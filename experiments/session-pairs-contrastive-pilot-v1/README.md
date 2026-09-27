# session-pairs-contrastive-pilot-v1

First paired ranking-loss pilot improved the answer margin; held-out pairs remained unsolved.

- Checkpoint: `checkpoints/session-pairs-contrastive-pilot-v1`
- Parent checkpoint: `checkpoints/session-pairs-full-v1`
- Source data: `.datasets_cache/session-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_delta_read": false, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T13:06:33.701000+00:00", "episodes": 100, "interrupted": false, "last_validation_loss": 0.9994009865654839, "loss_first_50": 3.8320869505405426, "loss_last_50": 3.8320869505405426, "objective": "answer_ce+pair_contrastive", "optimizer_steps": 50, "options": null, "pair_contrastive_weight": 16.0, "seed": 42, "started_at": "2026-09-26T13:05:39.861000+00:00", "steps": 50, "supervise_eos": null, "window_size": 256}`
