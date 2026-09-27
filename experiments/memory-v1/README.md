# memory-v1

Initial synthetic memory baseline; answer accuracy remained low despite the loss plateau.

Result: User-reported test: 9/100 exact answers overall; multi-hop 0/37, recall 4/34, update 5/29.

- Checkpoint: `checkpoints/memory-v1`
- Parent checkpoint: `none`
- Source data: `memory-v1`
- Adapter: `{"checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_hidden_size": 256}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T08:34:50.234000+00:00", "episodes": 1000, "interrupted": false, "last_validation_loss": 1.618173095703125, "loss_first_50": 3.2564439096450806, "loss_last_50": 1.8294042981900274, "objective": "answer_ce", "optimizer_steps": 1000, "options": null, "seed": 42, "started_at": "2026-09-26T07:46:19.670000+00:00", "steps": 1000, "supervise_eos": null, "window_size": 512}`
