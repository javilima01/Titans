# session-profile-smoke-23

Single-layer adapter smoke run at layer 23.

- Checkpoint: `checkpoints/session-profile-smoke-23`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-profile-v1`
- Adapter: `{"checkpoint_memory": true, "layer_indices": [23], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_hidden_size": 256}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T10:59:00.165000+00:00", "episodes": 16, "interrupted": false, "last_validation_loss": 4.250142317551833, "loss_first_50": 4.890792146021006, "loss_last_50": 4.890792146021006, "objective": "answer_ce", "optimizer_steps": 8, "options": null, "seed": 42, "started_at": "2026-09-26T10:56:47.375000+00:00", "steps": 8, "supervise_eos": null, "window_size": 512}`
