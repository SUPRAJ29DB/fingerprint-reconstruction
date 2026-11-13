from pathlib import Path
import random
import cv2
import numpy as np
import pandas as pd
from src.preprocessing.mask_generator import generate_forensic_mask_and_prompt

def main():
    raw_b = Path("datasets/raw/SD302b")
    mask_dir = Path("datasets/processed/masks")
    out_prompts = Path("datasets/annotations/prompts.csv")
    out_split = Path("datasets/annotations/split_manifest.csv")
    mask_dir.mkdir(parents=True, exist_ok=True)
    out_prompts.parent.mkdir(parents=True, exist_ok=True)

    b_files = list(raw_b.rglob("*.png"))
    if not b_files:
        print("[Error] No images in datasets/raw/SD302b. Run `python scripts/generate_pairs.py` first.")
        return

    records = []
    for idx, img_path in enumerate(b_files):
        split = "val" if (idx % 5 == 0) else "train"
        for v in range(2):
            ratio = random.choice([0.2, 0.4, 0.6])
            mask, prompt = generate_forensic_mask_and_prompt((256, 256), occlusion_ratio=ratio)
            mask_path = mask_dir / f"{img_path.stem}_mask_v{v}.png"
            cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
            records.append({
                "exemplar_path": str(img_path),
                "mask_path": str(mask_path),
                "occlusion_target": ratio,
                "prompt": prompt,
                "split": split,
            })

    df = pd.DataFrame(records)
    df.to_csv(out_prompts, index=False)
    df[["exemplar_path", "mask_path", "split"]].to_csv(out_split, index=False)
    print(f"[Success] Generated {len(df)} masked samples & forensic prompts -> {out_prompts}")

if __name__ == "__main__":
    main()
