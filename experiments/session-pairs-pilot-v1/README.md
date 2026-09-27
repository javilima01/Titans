# session-pairs-pilot-v1

First paired-dataset pilot.

- Checkpoint: `checkpoints/session-pairs-pilot-v1`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-pairs-256-v1`
- Adapter: `{"checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_hidden_size": 256}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T12:33:27.826000+00:00", "episodes": 100, "interrupted": true, "last_validation_loss": null, "loss_first_50": 3.913202923697394, "loss_last_50": 3.913202923697394, "objective": "answer_ce", "optimizer_steps": 18, "options": null, "seed": 42, "started_at": "2026-09-26T12:27:07.558000+00:00", "steps": 18, "supervise_eos": null, "window_size": 256}`
