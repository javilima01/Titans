# session-profile-batch2-smoke

Batch-size-two smoke run for cross-session training.

- Checkpoint: `checkpoints/session-profile-batch2-smoke`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-profile-256-v1`
- Adapter: `{"checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_hidden_size": 256}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T11:16:02.127000+00:00", "episodes": 16, "interrupted": true, "last_validation_loss": null, "loss_first_50": 3.3465723991394043, "loss_last_50": 3.3465723991394043, "objective": "answer_ce", "optimizer_steps": 2, "options": null, "seed": 42, "started_at": "2026-09-26T11:15:01.178000+00:00", "steps": 2, "supervise_eos": null, "window_size": 256}`
