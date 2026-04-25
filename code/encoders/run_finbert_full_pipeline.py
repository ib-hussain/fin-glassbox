from pathlib import Path
import gc
import json
import torch

from finbert_encoder import FinBERTConfig, FinBERTHyperparameterSearch, FinBERTMLMTrainer, FinBERTBaseEmbeddingExtractor, FinBERTPCAProjector, ensure_dir

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

    cfg.num_workers = 6
    cfg.batch_size = 24
    cfg.eval_batch_size = 64
    cfg.epochs = 3
    cfg.max_rows = None
    cfg.sample_frac = None
    cfg.sample_mode = "balanced-year"
    cfg.processor = "cuda"
    cfg.fp16 = True

    ensure_dir(cfg.models_path)
    ensure_dir(cfg.embeddings_path)
    ensure_dir(cfg.results_path)
    ensure_dir(cfg.code_results_path)

    print("========== FINBERT FULL PIPELINE ==========")
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
    print("Dataset:", cfg.dataset_csv)
    print("Models:", cfg.models_path)
    print("Embeddings:", cfg.embeddings_path)

    print("\n========== STAGE 1: HPO ON CHUNK 3 SAMPLE ==========")
    hpo_cfg = FinBERTConfig(**vars(cfg))
    hpo_cfg.max_rows = 30000
    hpo_cfg.epochs = 2
    hpo_cfg.num_workers = 6
    hpo_cfg.models_path = cfg.models_path / "hpo"
    hpo_cfg.results_path = cfg.results_path / "hpo"
    hpo_cfg.code_results_path = cfg.code_results_path
    search = FinBERTHyperparameterSearch(hpo_cfg, chunk_id=3, trials=12, study_name="finbert_mlm_chunk3_final")
    best = search.run()
    best_params = best.get("best_params", {})
    print("BEST PARAMS:", json.dumps(best_params, indent=2))

    print("\n========== STAGE 2: FULL TRAINING CHUNKS 1, 2, 3 ==========")
    train_cfg = FinBERTConfig(**vars(cfg))
    train_cfg = apply_best_params(train_cfg, best_params)
    train_cfg.max_rows = None
    train_cfg.sample_frac = None
    train_cfg.num_workers = 6
    train_cfg.save(train_cfg.code_results_path / "finbert_full_training_config.json")

    for chunk in [1, 2, 3]:
        print(f"\n========== TRAIN MLM CHUNK {chunk} ==========")
        trainer = FinBERTMLMTrainer(train_cfg, chunk_id=chunk)
        trainer.train(resume=True)
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n========== STAGE 3: EMBEDDING EXTRACTION + PCA PROJECTION ==========")
    embed_cfg = FinBERTConfig(**vars(train_cfg))
    embed_cfg.max_rows = None
    embed_cfg.sample_frac = None
    embed_cfg.eval_batch_size = 64
    embed_cfg.num_workers = 6
    embed_cfg.pca_batch_size = 4096

    for chunk in [1, 2, 3]:
        print(f"\n========== EMBED CHUNK {chunk}: 768 EXTRACTION ==========")
        extractor = FinBERTBaseEmbeddingExtractor(embed_cfg, chunk_id=chunk)
        extractor.extract_all_768(overwrite=False)

        print(f"\n========== EMBED CHUNK {chunk}: PCA FIT TRAIN ONLY ==========")
        projector = FinBERTPCAProjector(embed_cfg, chunk_id=chunk)
        projector.fit_train_pca(overwrite=False)

        print(f"\n========== EMBED CHUNK {chunk}: PCA TRANSFORM TRAIN/VAL/TEST ==========")
        projector.transform_all(overwrite=False)

        del extractor, projector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n========== FINBERT FULL PIPELINE COMPLETE ==========")
    print("Final models:", train_cfg.models_path)
    print("Final embeddings:", train_cfg.embeddings_path)

if __name__ == "__main__":
    main()


