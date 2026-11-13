from src.training.trainer import ForensicVLMTrainer

def main():
    print("=" * 60)
    print("Starting ForensicVLM Training on DGX Spark")
    print("=" * 60)
    trainer = ForensicVLMTrainer(config_path="configs/training.yaml")
    trainer.train()

if __name__ == "__main__":
    main()
