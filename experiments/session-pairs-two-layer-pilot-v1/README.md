# session-pairs-two-layer-pilot-v1

Two-layer adapter did not improve held-out paired recall in the pilot.

- Checkpoint: `checkpoints/session-pairs-two-layer-pilot-v1`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [11, 23], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_delta_read": false, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T12:53:36.309000+00:00", "episodes": 100, "interrupted": false, "last_validation_loss": 1.2757256825764973, "loss_first_50": 2.8629244502101625, "loss_last_50": 2.8629244502101625, "objective": "answer_ce", "optimizer_steps": 50, "options": null, "seed": 42, "started_at": "2026-09-26T12:52:24.332000+00:00", "steps": 50, "supervise_eos": null, "window_size": 256}`
