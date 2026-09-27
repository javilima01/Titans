# session-profile-smoke-11

Single-layer adapter smoke run at layer 11.

- Checkpoint: `checkpoints/session-profile-smoke-11`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-profile-v1`
- Adapter: `{"checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_hidden_size": 256}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T11:02:29.903000+00:00", "episodes": 16, "interrupted": false, "last_validation_loss": 4.235266465407151, "loss_first_50": 4.88262044167032, "loss_last_50": 4.88262044167032, "objective": "answer_ce", "optimizer_steps": 8, "options": null, "seed": 42, "started_at": "2026-09-26T10:59:35.507000+00:00", "steps": 8, "supervise_eos": null, "window_size": 512}`
