# session-pairs-repeat-mps-v1

Four more epochs on the same 50 pairs: zero fully correct held-out pairs.

- Checkpoint: `checkpoints/session-pairs-repeat-mps-v1`
- Parent checkpoint: `checkpoints/session-pairs-pilot-mps-v1`
- Source data: `.datasets_cache/session-pairs-256-v1`
- Adapter: `{"checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_hidden_size": 256}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T12:40:32.453000+00:00", "episodes": 100, "interrupted": false, "last_validation_loss": 1.354522493150499, "loss_first_50": 1.8598475945847375, "loss_last_50": 1.6593348304075854, "objective": "answer_ce", "optimizer_steps": 200, "options": null, "seed": 42, "started_at": "2026-09-26T12:37:15.286000+00:00", "steps": 200, "supervise_eos": null, "window_size": 256}`

Other reports:

- `checkpoints/session-pairs-repeat-mps-v1/paired-diagnostic-24.json`
