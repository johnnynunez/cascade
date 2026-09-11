# Segmented color regression samples

`segmented_color_samples.npz` contains unchanged uint8 **BGR** pixels from
real YOLOE prompt-free segmentation masks over two Isaac bridge camera frames.
It is recorded rendered-camera data, not hand-painted test images. The JSON
sidecar identifies each camera/candidate, detector label, original output and
expected color. Labels are provenance only; no label reaches the estimator.

Source evidence: `runs/delivery-20260911/perception-repair/baseline/`:

- `cam0.npz`, `side.npz`: full BGR, metric depth, K and camera transform.
- `cam0_open_masks.npz`, `side_open_masks.npz`: complete model masks.
- `report.json`: all raw candidates and photometric statistics, before changes.

Sampling is identical to `mask_color`: `ys,xs = np.nonzero(mask)`, then
`np.random.default_rng(0).choice(len(ys), 4000, replace=False)` when necessary.
The 4000 chosen BGR triplets are stored as `(4000,1,3)`; no rescaling, color
conversion, painting or pixel substitution occurs. All 13 prompt-free
candidates across the two views are retained, including neutral surfaces and
darkness, not only positive examples. The model was
`models/yoloe-11s-seg-pf.pt`, confidence 0.25, reached through the production
`OpenVocabDetector` parser. Raw detector names are not semantic ground truth.

The pink candidates (`cam0_2`, `side_4`) were incorrectly white at the original
S<45 achromatic boundary. Their full-mask HSV medians are [156,42,238] and
[155,37,241]. All other expected tags preserve the observed baseline behavior;
the bin straddles orange/yellow across views, so this fixture does not claim
an absolute material-color annotation for it.

`test_colors.py` checks these samples directly. Its selection tests rearrange
the same measured color samples into an explicitly **synthetic geometry** with
arbitrary shared detector labels and an orange distractor ranked first. Those
tests prove color selection/refusal, not live metric localization. Live and
same-frame real-model localization evidence is separate in the run directory.
