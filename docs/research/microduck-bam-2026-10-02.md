# MicroDuck BAM follow-up — 2 October 2026

[Isaac Lab PR #8161](https://github.com/isaac-sim/IsaacLab/pull/8161)
remains an open draft at `28aa1fca5843208ff9a67935695a4d5376e44d50`, the
revision already loaded by this candidate. Independent downloads of its three
runtime modules match the local manifest. Its distinction between raw torque
for battery sag and clamped applied torque for friction is already present.
Our explicit drive/joint limits are both 0.96 N·m. The 1.068 N·m figure derives
from a different voltage; it does not establish a reason to raise our limits.

The model remains frozen during evaluation. The
[Rhoban branch beyond Pollen's lock](https://github.com/Rhoban/bam/compare/62bd8ce12154340be97e06f7f41a0ca8f116d967...57d13ead53206a6bf0db3d66f86506ae8c2ce01a)
changes battery-sag estimation and units. A
[new XL330 fit](https://github.com/Rhoban/bam/blob/e9a619d56da5236206f4de6ceec2c1ee1b497b5c/bam/params/xl330/m6.json)
changes electrical parameters, friction and armature, and adds fitted delay.
The [maintainer explains a corrected identification-rig mass and new data](https://github.com/Rhoban/bam/issues/17#issuecomment-5678120069).
Neither change has demonstrated compatibility with the deployed checkpoint.

[Newton PR #4380's proposed API](https://github.com/newton-physics/newton/pull/4380#issuecomment-5916184925)
reduces the cost of publishing force properties. It remains unmerged and does
not replace load extraction. The
[public ovphysx interfaces examined](https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/read_write/readable.html)
do not establish the separate own-friction and joint-limit reactions needed
by the complete BAM load contract. This is an evidence gap for the examined
interfaces, not proof that extending PhysX is impossible.

This source research adds no simulation or locomotion result. Future changes
should first compare saturation, reconstructed load and measured progress with
the frozen recipe, and evaluate any new fit as a separate named recipe. The
[current native report](../MICRODUCK_CONTINUATION_20261002.md) records the
actual implementation's successes and failures.
