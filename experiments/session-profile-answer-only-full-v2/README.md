# session-profile-answer-only-full-v2

Full answer-only profile training run.

- Checkpoint: `checkpoints/session-profile-answer-only-full-v2`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-profile-answer-only-v2`
- Adapter: `{"checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_hidden_size": 256}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T12:05:19.605000+00:00", "episodes": 1200, "interrupted": true, "last_validation_loss": null, "loss_first_50": 3.076983694649913, "loss_last_50": 3.076983694649913, "objective": "answer_ce", "optimizer_steps": 40, "options": null, "seed": 42, "started_at": "2026-09-26T11:50:19.744000+00:00", "steps": 40, "supervise_eos": null, "window_size": 256}`
