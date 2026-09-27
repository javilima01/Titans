# session-pairs-delta-pilot-v1

Fast-weight-only delta read lowered CE but did not solve held-out pairs.

- Checkpoint: `checkpoints/session-pairs-delta-pilot-v1`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_delta_read": true, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T12:50:15.032000+00:00", "episodes": 100, "interrupted": false, "last_validation_loss": 1.1648797988891602, "loss_first_50": 3.134389522884573, "loss_last_50": 3.134389522884573, "objective": "answer_ce", "optimizer_steps": 50, "options": null, "seed": 42, "started_at": "2026-09-26T12:49:23.812000+00:00", "steps": 50, "supervise_eos": null, "window_size": 256}`
