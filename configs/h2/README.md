# Unitree H2 policy bundle pins

`bundle.json` pins NVIDIA's public Isaac Sim bundle for the Unitree H2
(`Velocity-H2-History-v0`): the exact asset-root paths and SHA-256 digests of
`policy.pt`, `env.yaml`, `IO_descriptors.yaml` and the `H2.usda` stub. The two
YAML files are vendored verbatim under `assets/h2/bundle/` (Apache-2.0, SPDX
headers kept; outside `configs/` because they are NVIDIA training exports, not
CASCADE profiles) so the policy contract can be parsed and tested offline; `policy.pt` and the USD are
never committed: `scripts/h2_assets.py` fetches the policy into
`runs/.install-cache/h2/` and refuses any byte that does not match the pin, and
the owner opens the USD from the asset root.

`src/cascade/control/h2_policy_contract.py` turns the two YAML files into the
contract the owner must honour (joint order, observation layout, action decode,
PD gains, fall criteria). It fails closed on any drift from the pins or from the
model's 255 → 14 shape. Design and admission status: `docs/HUMANOID_H2.md`.
