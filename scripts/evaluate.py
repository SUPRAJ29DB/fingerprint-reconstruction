from pathlib import Path
from experiments.run_evaluation import run_full_benchmark

def main():
    ckpt = "checkpoints/best_model/forensic_vlm_best_stage1.pt"
    if not Path(ckpt).exists():
        print(f"[Error] Checkpoint not found at {ckpt}. Run `python scripts/train.py` first.")
        return
    run_full_benchmark(
        checkpoint_path=ckpt,
        test_csv="datasets/annotations/prompts.csv",
        stage="stage1",
        output_report_csv="reports/metrics/benchmark_results.csv",
    )

if __name__ == "__main__":
    main()
