# Vendored NVIDIA H2 policy bundle exports

`env.yaml` and `IO_descriptors.yaml` are byte-identical copies of NVIDIA's
public Isaac Sim files `Isaac/Samples/Policies/h2/{env.yaml,IO_descriptors.yaml}`
(Apache-2.0, SPDX headers kept; identical on the 6.1 and 6.2 asset roots),
pinned by SHA-256 in `configs/h2/bundle.json` and parsed by
`src/cascade/control/h2_policy_contract.py`. They are training exports, not
CASCADE profiles (hence outside `configs/`): never edit them — a changed byte
fails the pin. `policy.pt` is not vendored (`scripts/h2_assets.py` fetches and
verifies it). Design: `docs/HUMANOID_H2.md`.
