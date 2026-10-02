# Factory M20 loose nut and bolt

These custom assets are from NVIDIA IsaacGymEnvs commit
`aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3`. The retained
`factory-acknowledgments.txt` explicitly places the nut and bolt assets under
the repository BSD 3-Clause `LICENSE.txt`. The two OBJ files are fetched with
`python scripts/fetch_factory_assets.py`; `manifest.json` pins their exact
upstream paths, byte lengths and SHA-256 hashes. Meshes are not stored in Git.

The retained upstream dimensions file specifies a 2.5 mm thread pitch, 30 mm
nut width across flats, 16 mm nut height, and 20 mm bolt nominal diameter.
Newton's Apache-2.0 `examples/contacts/example_nut_bolt_sdf.py` supplies the
reference SDF parameters. The CASCADE experiment adds a SO-101-mounted powered
socket and independent pose/contact verification; it does not certify seating
torque, preload or hardware behavior.
