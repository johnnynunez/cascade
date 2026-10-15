# models/

The SAM weights live here: `configs/demo.yaml` loads them by absolute path (`segmenter.model: ${repo}/models/sam2.1_l.pt`), from any working directory.

- Get them: `scripts/install.sh --profile spark` downloads `sam2.1_l.pt` (SAM 2.1-L) here and checks its size.
- They are gitignored (`models/sam*.pt`); this file keeps the directory in every checkout.
