# session-pairs-clip10-pilot-v1

Inner gradient norm cap of 10 destabilized validation loss.

- Checkpoint: `checkpoints/session-pairs-clip10-pilot-v1`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 10.0, "memory_chunk_size": 16, "memory_delta_read": false, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T12:52:07.013000+00:00", "episodes": 100, "interrupted": false, "last_validation_loss": 12.639354281955296, "loss_first_50": 6.582986580474036, "loss_last_50": 6.582986580474036, "objective": "answer_ce", "optimizer_steps": 50, "options": null, "seed": 42, "started_at": "2026-09-26T12:51:15.603000+00:00", "steps": 50, "supervise_eos": null, "window_size": 256}`
