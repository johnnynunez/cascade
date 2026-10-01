"""Letterbox coordinates must match the depth frame on CPU and CUDA."""
from types import SimpleNamespace
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from cascade.perception.detector import OpenVocabDetector
from cascade.types import Frame


def parse(raw, shape, monkeypatch, device='cpu'):
    torch = pytest.importorskip('torch')
    cuda = device.startswith('cuda')
    if cuda and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    monkeypatch.setenv('CASCADE_REQUIRE_CUDA', '1' if cuda else '0')
    masks = torch.as_tensor(raw, device=device)[None]
    boxes = SimpleNamespace(cls=torch.tensor([0], device=device),
                            conf=torch.tensor([.9], device=device),
                            xyxy=torch.tensor([[0, 0, shape[1], shape[0]]], device=device))
    class Boxes:
        def __len__(self):
            return 1
    wrapped = Boxes()
    wrapped.__dict__.update(vars(boxes))
    result = SimpleNamespace(names={0: 'object'}, boxes=wrapped,
                             masks=SimpleNamespace(data=masks), orig_shape=shape)
    frame = Frame(rgb=np.zeros((*shape, 3), np.uint8), depth_m=None, K=np.eye(3))
    detector = object.__new__(OpenVocabDetector)
    mask = detector._parse(result, frame)[0].mask
    if cuda:
        assert mask.device.type == 'cuda'
        assert mask.dtype == torch.bool
        return mask.cpu().numpy()
    assert isinstance(mask, np.ndarray) and mask.dtype == np.bool_
    return mask


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_native_720_letterbox_matches_unpadded_depth_pixels(monkeypatch, device):
    raw = np.zeros((384, 640), np.uint8)
    raw[112:122, 100:110] = 1  # content y=100..109 after removing 12px top padding
    got = parse(raw, (720, 1280), monkeypatch, device)
    expected = np.zeros((720, 1280), bool)
    expected[200:220, 200:220] = True
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('transpose', [False, True])
def test_odd_padding_extra_pixel_is_removed_from_end(monkeypatch, device, transpose):
    raw = np.zeros((8, 8), np.uint8)
    raw[2:4, 3:5] = 1
    raw[0, :] = raw[6:, :] = 1  # padding must never become foreground
    expected = np.zeros((5, 8), bool)
    expected[1:3, 3:5] = True
    if transpose:
        raw, expected = raw.T.copy(), expected.T.copy()
    got = parse(raw, expected.shape, monkeypatch, device)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('shape', [(8, 8), (6, 10)])
def test_native_resolution_retina_masks_are_unchanged(monkeypatch, device, shape):
    raw = np.zeros(shape, np.uint8)
    raw[1:3, 2:5] = 1
    np.testing.assert_array_equal(parse(raw, shape, monkeypatch, device), raw.astype(bool))


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_same_as_official_scale_masks_on_native_canvas(monkeypatch, device):
    torch = pytest.importorskip('torch')
    ops = pytest.importorskip('ultralytics.utils.ops')
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    raw = np.random.default_rng(4).integers(0, 2, (384, 640), dtype=np.uint8)
    tensor = torch.as_tensor(raw, device=device)[None, None]
    try:
        expected = ops.scale_masks(tensor, (720, 1280), mode='nearest')[0, 0] > .5
    except TypeError:
        pytest.skip('Installed Ultralytics lacks nearest scale_masks; explicit geometry tests still apply')
    np.testing.assert_array_equal(parse(raw, (720, 1280), monkeypatch, device), expected.cpu().numpy())


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('label', ['green cube', 'orange'])
def test_saved_native_masks_match_official_cuda_replay(monkeypatch, device, label):
    root = Path(__file__).parent / 'fixtures/detector_letterbox'
    receipt = json.loads((root / 'receipt.json').read_text())
    row = next(r for r in receipt['rows'] if r['label'] == label)
    path = root / row['npz']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    assert row['old_vs_historical_different_pixels'] == 0
    with np.load(path, allow_pickle=False) as data:
        got = parse(data['raw_mask'], (720, 1280), monkeypatch, device)
        np.testing.assert_array_equal(got, data['corrected_mask'])
        assert not np.array_equal(got, data['historical_mask'])
