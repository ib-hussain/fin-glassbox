from pathlib import Path
import gc
import json
import torch

from finbert_encoder import FinBERTConfig, FinBERTMLMTrainer, FinBERTBaseEmbeddingExtractor, FinBERTPCAProjector, ensure_dir

def apply_best_params(cfg, best_params):
    cfg.learning_rate = float(best_params.get("learning_rate", cfg.learning_rate))
    cfg.weight_decay = float(best_params.get("weight_decay", cfg.weight_decay))
    cfg.warmup_ratio = float(best_params.get("warmup_ratio", cfg.warmup_ratio))
    cfg.batch_size = int(best_params.get("batch_size", cfg.batch_size))
    cfg.eval_batch_size = min(96, max(32, cfg.batch_size * 2))
    cfg.gradient_accumulation_steps = int(best_params.get("gradient_accumulation_steps", cfg.gradient_accumulation_steps))
    cfg.mlm_probability = float(best_params.get("mlm_probability", cfg.mlm_probability))
    cfg.epochs = max(3, int(best_params.get("epochs", 3)))
    return cfg

def main():
    cfg = FinBERTConfig()
    cfg.repo_root = Path(".").resolve()
    cfg.env_file = Path(".env")
    cfg.dataset_csv = Path("final/filings_finbert_chunks_balanced_25y_cap40000.csv")
    cfg.resolve()

    cfg.processor = "cuda"
    cfg.fp16 = True
    cfg.num_workers = 6
    cfg.max_rows = None
    cfg.sample_frac = None
    cfg.sample_mode = "balanced-year"

    ensure_dir(cfg.models_path)
    ensure_dir(cfg.embeddings_path)
    ensure_dir(cfg.results_path)
    ensure_dir(cfg.code_results_path)

    best_path = cfg.code_results_path / "hpo" / "finbert_mlm_chunk3_final_best_params.json"
    if not best_path.exists():
        raise FileNotFoundError(f"HPO best params not found: {best_path}")

    best = json.loads(best_path.read_text())
    best_params = best.get("best_params", {})
    cfg = apply_best_params(cfg, best_params)
    cfg.save(cfg.code_results_path / "finbert_resume_training_config.json")

    print("========== FINBERT RESUME AFTER HPO ==========")
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
    print("Best params:", json.dumps(best_params, indent=2))
    print("Final training config:", json.dumps(cfg.serializable(), indent=2))

    for chunk in [1, 2, 3]:
        print(f"\n========== TRAIN/RESUME MLM CHUNK {chunk} ==========")
        trainer = FinBERTMLMTrainer(cfg, chunk_id=chunk)
        trainer.train(resume=True)
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n========== EMBEDDING EXTRACTION + PCA ==========")
    cfg.eval_batch_size = 64
    cfg.num_workers = 6
    cfg.pca_batch_size = 4096

    for chunk in [1, 2, 3]:
        print(f"\n========== EMBED CHUNK {chunk}: 768 EXTRACTION ==========")
        extractor = FinBERTBaseEmbeddingExtractor(cfg, chunk_id=chunk)
        extractor.extract_all_768(overwrite=False)

        print(f"\n========== EMBED CHUNK {chunk}: PCA FIT TRAIN ONLY ==========")
        projector = FinBERTPCAProjector(cfg, chunk_id=chunk)
        projector.fit_train_pca(overwrite=False)

        print(f"\n========== EMBED CHUNK {chunk}: PCA TRANSFORM ALL ==========")
        projector.transform_all(overwrite=False)

        del extractor, projector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n========== FINBERT RESUME PIPELINE COMPLETE ==========")
    print("Models:", cfg.models_path)
    print("Embeddings:", cfg.embeddings_path)

if __name__ == "__main__":
    main()


