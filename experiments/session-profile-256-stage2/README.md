# session-profile-256-stage2

Second stage of the cross-session profile run.

- Checkpoint: `checkpoints/session-profile-256-stage2`
- Parent checkpoint: `checkpoints/session-profile-256-strong-v1`
- Source data: `.datasets_cache/session-profile-256-stage2`
- Adapter: `{"checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_hidden_size": 256}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T11:38:36.280000+00:00", "episodes": 1000, "interrupted": true, "last_validation_loss": null, "loss_first_50": 1.81397924752071, "loss_last_50": 1.81397924752071, "objective": "answer_ce", "optimizer_steps": 5, "options": null, "seed": 42, "started_at": "2026-09-26T11:35:04.079000+00:00", "steps": 5, "supervise_eos": null, "window_size": 256}`
