# session-pairs-full-v1

Full 600-pair answer-only training learned common answer tokens but did not use memory reliably.

Result: Pilot validation: 1/24 exact answers, zero fully correct pairs.

- Checkpoint: `checkpoints/session-pairs-full-v1`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_delta_read": false, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T13:03:36.780000+00:00", "episodes": 1200, "interrupted": false, "last_validation_loss": 1.0700627432929144, "loss_first_50": 2.5948058888316154, "loss_last_50": 1.4861922274648616, "objective": "answer_ce", "optimizer_steps": 600, "options": null, "seed": 42, "started_at": "2026-09-26T12:54:22.564000+00:00", "steps": 600, "supervise_eos": null, "window_size": 256}`

## Evaluations

- `validation-24.json` (validation):
  - normal: EM 0.041666666666666664, F1 0.38257575757575757, both pairs 0/12
  - disabled: EM 0.0, F1 0.0, both pairs 0/12
  - reset: EM 0.0, F1 0.3055555555555555, both pairs 0/12
