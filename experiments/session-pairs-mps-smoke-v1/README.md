# session-pairs-mps-smoke-v1

Apple MPS training smoke run.

- Checkpoint: `checkpoints/session-pairs-mps-smoke-v1`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-pairs-256-v1`
- Adapter: `{"checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_hidden_size": 256}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T12:33:57.184000+00:00", "episodes": 4, "interrupted": false, "last_validation_loss": 9.917769432067871, "loss_first_50": 4.944236437479655, "loss_last_50": 4.944236437479655, "objective": "answer_ce", "optimizer_steps": 2, "options": null, "seed": 42, "started_at": "2026-09-26T12:33:45.740000+00:00", "steps": 2, "supervise_eos": null, "window_size": 256}`
