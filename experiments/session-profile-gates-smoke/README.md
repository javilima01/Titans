# session-profile-gates-smoke

Gate initialization smoke run.

- Checkpoint: `checkpoints/session-profile-gates-smoke`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-profile-256-v1`
- Adapter: `{"checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_hidden_size": 256}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T11:13:08.085000+00:00", "episodes": 16, "interrupted": false, "last_validation_loss": 2.4523444797681724, "loss_first_50": 5.238682104616749, "loss_last_50": 5.238682104616749, "objective": "answer_ce", "optimizer_steps": 4, "options": null, "seed": 42, "started_at": "2026-09-26T11:11:12.841000+00:00", "steps": 4, "supervise_eos": null, "window_size": 256}`
