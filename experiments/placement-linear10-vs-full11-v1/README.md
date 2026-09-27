# placement-linear10-vs-full11-v1

Matched placement ablation: Titans on Qwen linear layer 10 versus full-attention layer 11, both from fresh initialization.

Result: Linear 10: familiar 123/160, 59/80 pairs; novel 1/160, 0 pairs. Full 11: familiar 117/160, 55/80 pairs; novel 0/160, 0 pairs. Placement helps familiar templates modestly and does not solve open-value recall.

- Checkpoint: `checkpoints/session-broad-linear10-v1`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-broad-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [10], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_delta_read": false, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-27T07:49:33.278000+00:00", "episodes": 1400, "interrupted": false, "last_validation_loss": 0.20531605451534957, "loss_first_50": 4.366112994559018, "loss_last_50": 0.3271769585990114, "objective": "answer_ce+pair_contrastive", "optimizer_steps": 700, "options": {"batch_size": 2, "checkpoint_decoder": false, "device": "mps", "dtype": null, "epochs": 1, "gradient_accumulation_steps": 1, "limit": null, "loss_chunk_size": 128, "lr": 0.0003, "max_grad_norm": 1.0, "max_steps": null, "pair_contrastive_weight": 16.0, "shuffle": false, "supervise_eos": true, "validation_limit": 40, "weight_decay": 0.01}, "pair_contrastive_weight": 16.0, "seed": 42, "started_at": "2026-09-27T07:38:49.478000+00:00", "steps": 700, "supervise_eos": true, "window_size": 256}`

Other reports:

- `experiments/placement-linear10-vs-full11-v1/summary.json`
