
from pathlib import Path

PROJECT_NAME = "ForensicVLM"

# All project directories
DIRECTORIES = [
    "configs",

    "datasets/raw/SD302e/original",
    "datasets/raw/SD302b/original",

    "datasets/processed/SD302e/images",
    "datasets/processed/SD302e/degraded",
    "datasets/processed/SD302e/masks",
    "datasets/processed/SD302e/metadata",

    "datasets/processed/SD302b/images",
    "datasets/processed/SD302b/metadata",

    "datasets/annotations",

    "src/inspection",
    "src/preprocessing",
    "src/pairing",
    "src/vlm",
    "src/reconstruction",
    "src/training",
    "src/evaluation",
    "src/visualization",
    "src/utils",

    "experiments/baseline_vision_only",
    "experiments/baseline_vlm",

    "checkpoints/best_model",
    "checkpoints/latest",

    "reports/figures",
    "reports/metrics",

    "scripts",
    "tests",
]

# Python files to create
PYTHON_FILES = [
    "src/__init__.py",

    "src/inspection/__init__.py",
    "src/inspection/inventory.py",
    "src/inspection/image_statistics.py",
    "src/inspection/visualize_samples.py",

    "src/preprocessing/__init__.py",
    "src/preprocessing/image_loader.py",
    "src/preprocessing/denoising.py",
    "src/preprocessing/normalization.py",
    "src/preprocessing/resizing.py",
    "src/preprocessing/mask_generator.py",

    "src/pairing/__init__.py",
    "src/pairing/synthetic_pairs.py",
    "src/pairing/ebts_parser.py",

    "src/vlm/__init__.py",
    "src/vlm/clip_encoder.py",
    "src/vlm/vision_encoder.py",
    "src/vlm/text_encoder.py",
    "src/vlm/fusion.py",
    "src/vlm/vlm_model.py",

    "src/reconstruction/__init__.py",
    "src/reconstruction/decoder.py",
    "src/reconstruction/unet.py",
    "src/reconstruction/losses.py",

    "src/training/__init__.py",
    "src/training/dataset.py",
    "src/training/trainer.py",
    "src/training/validate.py",
    "src/training/checkpoint.py",

    "src/evaluation/__init__.py",
    "src/evaluation/metrics.py",
    "src/evaluation/ridge_analysis.py",
    "src/evaluation/minutiae_analysis.py",
    "src/evaluation/uncertainty.py",
    "src/evaluation/compare_baselines.py",

    "src/visualization/__init__.py",
    "src/visualization/reconstruction_grid.py",
    "src/visualization/training_curves.py",

    "src/utils/__init__.py",
    "src/utils/seed.py",
    "src/utils/device.py",
    "src/utils/logger.py",

    "scripts/inspect_dataset.py",
    "scripts/preprocess_dataset.py",
    "scripts/generate_pairs.py",
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/predict.py",

    "tests/test_dataset.py",
    "tests/test_model.py",
    "tests/test_masks.py",
    "tests/test_metrics.py",
]

# Configuration and documentation templates
FILES = {
    "README.md": """# ForensicVLM

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
""",

    "requirements.txt": """transformers
torchvision
Pillow
numpy
pandas
opencv-python
scikit-image
matplotlib
tqdm
PyYAML
pytest
""",

    ".gitignore": """__pycache__/
*.py[cod]
.venv/
.env
.ipynb_checkpoints/
checkpoints/
experiments/
datasets/raw/
datasets/processed/
""",

    "configs/dataset.yaml": """dataset:
  sd302e: datasets/raw/SD302e/original
  sd302b: datasets/raw/SD302b/original
  processed: datasets/processed
  image_size: 224
  train_ratio: 0.8
  validation_ratio: 0.1
  test_ratio: 0.1
""",

    "configs/model.yaml": """model:
  name: ForensicVLM
  clip_model: openai/clip-vit-base-patch16
  image_size: 224
  hidden_dim: 768
  attention_heads: 8
  decoder: unet
""",

    "configs/training.yaml": """training:
  batch_size: 8
  epochs: 10
  learning_rate: 0.0001
  seed: 42
  checkpoint_dir: checkpoints
  device: cuda
""",

    "datasets/annotations/training_pairs.csv":
        "input_path,target_path,mask_path,split\n",

    "datasets/annotations/prompts.csv":
        "prompt_id,prompt\n",

    "datasets/annotations/split_manifest.csv":
        "image_path,split,group_id,group_id_verified\n",

    "datasets/processed/SD302e/metadata/image_inventory.csv":
        "path,width,height,mode,format\n",

    "datasets/processed/SD302b/metadata/image_inventory.csv":
        "path,width,height,mode,format\n",

    "reports/experiment_summary.md":
        "# ForensicVLM Experiment Summary\n\n"
        "Record methodology, experiments, limitations and results here.\n",
}


def main():
    root = Path(PROJECT_NAME)

    if root.exists():
        print(f"Project directory already exists: {root.resolve()}")
        print("Existing files will not be overwritten.")
    else:
        root.mkdir(parents=True)
        print(f"Created project: {root.resolve()}")

    # Create directories
    for directory in DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)

    # Create Python files without overwriting existing work
    for filename in PYTHON_FILES:
        path = root / filename
        if not path.exists():
            path.write_text(
                '"""ForensicVLM module."""\n',
                encoding="utf-8"
            )

    # Create documentation and configuration files
    for filename, content in FILES.items():
        path = root / filename
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    print("\nForensicVLM structure created successfully!")
    print(f"Project location: {root.resolve()}")
    print(f"Python modules: {len(PYTHON_FILES)}")
    print(f"Configuration/documentation files: {len(FILES)}")


if __name__ == "__main__":
    main()
