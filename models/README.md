# models/

The robot server runs with this directory as its working directory, so the SAM weights it loads by bare name resolve here (`segmenter.model: sam2.1_l.pt` in `configs/demo.yaml`).

- Get them: `scripts/install.sh --profile spark` downloads `sam2.1_l.pt` (SAM 2.1-L) here and checks its size.
- They are gitignored (`models/sam*.pt`); this file keeps the directory in every checkout.
