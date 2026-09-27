# session-paired-overfit-v1

Two-example counterfactual overfit: the adapter learned the distinction after repeated steps.

- Checkpoint: `checkpoints/session-paired-overfit-v1`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-paired-probe-v1`
- Adapter: `{"checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_hidden_size": 256}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T12:09:41.513000+00:00", "episodes": 2, "interrupted": true, "last_validation_loss": null, "loss_first_50": 0.6003405968389975, "loss_last_50": 0.6003405968389975, "objective": "answer_ce", "optimizer_steps": 26, "options": null, "seed": 42, "started_at": "2026-09-26T12:05:31.007000+00:00", "steps": 26, "supervise_eos": null, "window_size": 256}`
