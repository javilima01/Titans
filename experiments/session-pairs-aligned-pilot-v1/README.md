# session-pairs-aligned-pilot-v1

Aligned query/key initialization improved a diagnostic margin but solved no held-out pairs.

- Checkpoint: `checkpoints/session-pairs-aligned-pilot-v1`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T12:46:29.412000+00:00", "episodes": 100, "interrupted": false, "last_validation_loss": 1.421441290113661, "loss_first_50": 2.5948058888316154, "loss_last_50": 2.5948058888316154, "objective": "answer_ce", "optimizer_steps": 50, "options": null, "seed": 42, "started_at": "2026-09-26T12:45:38.717000+00:00", "steps": 50, "supervise_eos": null, "window_size": 256}`
