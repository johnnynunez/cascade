# Align detector masks with their RGB-D frame

The YOLOE result supplies boxes in original image coordinates, but its default
segmentation masks remain on the letterboxed inference canvas. Resizing that
whole canvas shifted the mask relative to RGB and depth. For the retained
1280×720 frames, the CUDA mask was 640×384 with 12 padded rows at each end.
The parser now removes the padding before nearest-neighbor resizing, on the
existing CPU or CUDA device. Native-resolution masks remain unchanged.

An offline replay on Spark used the same source, weights, RGB-D and production
query resolver as native launch05. It reproduced both historical target masks
exactly (zero differing pixels). Ultralytics 8.4.170 `scale_masks`, with nearest
interpolation on CUDA, then changed 584 pixels for green and 1,102 for orange.
The implementation matches its content bounds directly so older supported
Ultralytics versions need not expose the newer interpolation argument.

The [replay receipt](../tests/fixtures/detector_letterbox/receipt.json) includes
versions, model hashes, raw tensor shape/device, padding, query, input hashes,
and hashes for two compact raw-mask fixtures. Tests compare those exact masks
with the recorded official CUDA output. Separate synthetic tests cover vertical,
horizontal and odd padding, preservation of native-resolution masks, and CPU/CUDA
agreement. No simulator, robot runtime or actuator was used for this replay.

This corrects image coordinates, not segmentation completeness. On the retained
grasp poses, a hypothetical closing-sweep check had classified 131 green-object
surface points and 306 orange-object points as outside their masks. Correct
alignment reduces these to four and zero respectively. One observed pink-cube
point still intersects the orange grasp's closing envelope. The four remaining
green boundary pixels are not erased or assigned a special tolerance.

The mask change also changes localization and future grasp requests. Historical
grasp replay is therefore diagnostic, not acceptance of the corrected perception
or of a closing guard. The currently published open-finger gate, motion limits,
and closing behavior are unchanged by this patch.
