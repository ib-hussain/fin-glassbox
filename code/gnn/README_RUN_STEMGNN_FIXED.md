# Fixed StemGNN Contagion Module — Drop-in Replacement

## What was fixed

1. HPO no longer writes trial checkpoints into `outputs/models/StemGNN/chunkN/`, so final training will not resume from a random HPO trial.
2. HPO uses `num_workers=0` and `persistent_workers=False` by default to prevent `OSError: [Errno 24] Too many open files`.
3. DataLoader workers are explicitly shut down after training, validation, prediction, XAI, and HPO trials.
4. Resume is architecture-safe: `window_size`, `multi_layer`, `stack_cnt`, horizons, and node count are checked before loading a checkpoint.
5. `--fresh` was added to `train-best` to archive stale checkpoints and start cleanly.
6. XAI is returned by high-level functions and optionally saved to disk. Level 1 and Level 2 XAI always run during prediction; GNNExplainer is opt-in.
7. Added `smoke` command for synthetic and real-data tests.
