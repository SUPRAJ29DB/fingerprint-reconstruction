# ForensicVLM

Vision-Language Model based experimental fingerprint image reconstruction.

## Datasets
- NIST SD 302e: Primary latent fingerprint dataset.
- NIST SD 302b: Separate exemplar analysis dataset.

## Architecture
CLIP Vision Encoder + CLIP Text Encoder
-> Cross-Attention Fusion
-> CNN/U-Net Decoder
-> Reconstruction Hypothesis

## Important
Synthetic reconstruction experiments use SD302e images.
SD302b is not treated as pixel-aligned ground truth for SD302e.
Generated regions are hypotheses, not forensic evidence.
