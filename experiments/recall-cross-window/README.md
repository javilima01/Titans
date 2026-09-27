# recall-cross-window

Cross-window recall baseline; all three memory modes scored zero on 100 validation examples.

Result: Validation: 0/100 exact answers for normal, disabled, and reset memory.

- Checkpoint: `checkpoints/recall-cross-window`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/recall-cross-window`
- Adapter: `{"checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_hidden_size": 256}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T09:57:33.509000+00:00", "episodes": 2000, "interrupted": false, "last_validation_loss": 1.82007763671875, "loss_first_50": 1.834236743927002, "loss_last_50": 1.8245778350830077, "objective": "answer_ce", "optimizer_steps": 1000, "options": null, "seed": 42, "started_at": "2026-09-26T09:09:46.435000+00:00", "steps": 1000, "supervise_eos": null, "window_size": 512}`

## Evaluations

- `recall-cross-window-validation.json` (validation):
  - normal: EM 0.0, F1 0.0, both pairs n/a/n/a
  - disabled: EM 0.0, F1 0.0, both pairs n/a/n/a
  - reset: EM 0.0, F1 0.0, both pairs n/a/n/a
