from pathlib import Path

def main():
    raw_b = Path("datasets/raw/SD302b")
    raw_e = Path("datasets/raw/SD302e")
    b_files = list(raw_b.rglob("*.png")) + list(raw_b.rglob("*.jpg"))
    e_files = list(raw_e.rglob("*.png")) + list(raw_e.rglob("*.jpg"))

    print("=" * 60)
    print("ForensicVLM Dataset Inspection Report")
    print("=" * 60)
    print(f"SD302b (Exemplars) path : {raw_b.resolve()} -> {len(b_files)} images found")
    print(f"SD302e (Latents)   path : {raw_e.resolve()} -> {len(e_files)} images found")

    if len(b_files) == 0 and len(e_files) == 0:
        print("\n[NOTICE] No images found in datasets/raw/SD302b or SD302e yet!")
        print("         Generate synthetic demo images or link your NIST folders before training.")
    else:
        if b_files:
            print(f"Sample SD302b file      : {b_files[0].name}")
        if e_files:
            print(f"Sample SD302e file      : {e_files[0].name}")
    print("=" * 60)

if __name__ == "__main__":
    main()
