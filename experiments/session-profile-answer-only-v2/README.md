# session-profile-answer-only-v2

Answer-only objective removed EOS-dominated supervision.

- Checkpoint: `checkpoints/session-profile-answer-only-v2`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-profile-answer-only-v2`
- Adapter: `{"checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_hidden_size": 256}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T11:44:09.903000+00:00", "episodes": 1200, "interrupted": true, "last_validation_loss": null, "loss_first_50": 3.3304630426260142, "loss_last_50": 3.3304630426260142, "objective": "answer_ce", "optimizer_steps": 6, "options": null, "seed": 42, "started_at": "2026-09-26T11:40:17.999000+00:00", "steps": 6, "supervise_eos": null, "window_size": 256}`
