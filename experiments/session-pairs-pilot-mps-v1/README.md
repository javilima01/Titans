# session-pairs-pilot-mps-v1

One layer, 50 pairs, one pass: zero fully correct validation pairs.

- Checkpoint: `checkpoints/session-pairs-pilot-mps-v1`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-pairs-256-v1`
- Adapter: `{"checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_hidden_size": 256}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T12:35:06.544000+00:00", "episodes": 100, "interrupted": false, "last_validation_loss": 1.325050459967719, "loss_first_50": 2.7594300157257488, "loss_last_50": 2.7594300157257488, "objective": "answer_ce", "optimizer_steps": 50, "options": null, "seed": 42, "started_at": "2026-09-26T12:34:13.293000+00:00", "steps": 50, "supervise_eos": null, "window_size": 256}`

## Evaluations

- `validation-24.json` (validation):
  - normal: EM 0.0, F1 0.46428571428571425, both pairs 0/12
  - disabled: EM 0.0, F1 0.0, both pairs 0/12
  - reset: EM 0.0, F1 0.45039682539682535, both pairs 0/12
