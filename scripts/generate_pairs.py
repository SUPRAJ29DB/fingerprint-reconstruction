from pathlib import Path
import cv2
import numpy as np
import pandas as pd

FRGP_NAMES = {
    "01": "right thumb", "02": "right index finger", "03": "right middle finger",
    "04": "right ring finger", "05": "right little finger", "06": "left thumb",
    "07": "left index finger", "08": "left middle finger", "09": "left ring finger",
    "10": "left little finger",
}

def parse_filename(p: Path) -> dict:
    parts = p.stem.split("_")
    subj = parts[0] if len(parts) > 0 else "00000001"
    frgp = parts[-1].zfill(2) if len(parts) > 1 else "01"
    return {
        "filepath": str(p),
        "filename": p.name,
        "subject_id": subj,
        "frgp": frgp,
        "finger_name": FRGP_NAMES.get(frgp, "fingerprint"),
    }

def create_synthetic_fallback_if_empty(raw_b: Path, raw_e: Path, count: int = 20):
    """Creates synthetic ridge pairs if raw NIST folders haven't been linked yet so pipeline can be tested."""
    raw_b.mkdir(parents=True, exist_ok=True)
    raw_e.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[0:256, 0:256]
    for i in range(1, count + 1):
        subj = f"0000{i:04d}"
        frgp = f"{(i % 10) + 1:02d}"
        freq = 0.30 + (i % 5) * 0.03
        base = (0.5 + 0.5 * np.sin((x * freq) + 3.5 * np.sin(y * 0.04))).astype(np.float32)
        b_path = raw_b / f"{subj}_302b_rolled_{frgp}.png"
        e_path = raw_e / f"{subj}_302e_latent_{frgp}.png"
        cv2.imwrite(str(b_path), (base * 255).astype(np.uint8))
        noise = np.random.normal(0, 0.1, base.shape).astype(np.float32)
        latent = np.clip(base * 0.8 + noise, 0, 1)
        cv2.imwrite(str(e_path), (latent * 255).astype(np.uint8))
    print(f"[Fallback] Created {count} synthetic exemplar/latent pairs in datasets/raw/SD302b & SD302e.")

def main():
    raw_b = Path("datasets/raw/SD302b")
    raw_e = Path("datasets/raw/SD302e")
    b_files = list(raw_b.rglob("*.png"))
    e_files = list(raw_e.rglob("*.png"))

    if not b_files:
        print("[Info] datasets/raw/SD302b is empty. Generating 20 synthetic fingerprint pairs for testing...")
        create_synthetic_fallback_if_empty(raw_b, raw_e, count=20)
        b_files = list(raw_b.rglob("*.png"))
        e_files = list(raw_e.rglob("*.png"))

    df_b = pd.DataFrame([parse_filename(p) for p in b_files])
    df_e = pd.DataFrame([parse_filename(p) for p in e_files]) if e_files else df_b.copy()

    paired = pd.merge(df_e, df_b, on=["subject_id", "frgp", "finger_name"], suffixes=("_latent", "_exemplar"))
    out_csv = Path("datasets/annotations/training_pairs.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out_csv, index=False)
    print(f"[Success] Generated {len(paired)} pairs -> {out_csv}")

if __name__ == "__main__":
    main()
