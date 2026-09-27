# session-broad-all24-scaled-smoke-v1

Two-step all-24-layer startup diagnostic with the scaled output gate logit -5.3.

Result: Losses 7.39 and 6.37 on the same first two pairs as the unscaled run; used this gate setting for the 200-step pilot.

- Checkpoint: `checkpoints/session-broad-all24-scaled-smoke-v1`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-broad-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_delta_read": false, "memory_gate_init": -5.3, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-27T08:56:46.046000+00:00", "episodes": 8, "interrupted": false, "last_validation_loss": null, "loss_first_50": 7.100695746285575, "loss_last_50": 7.100695746285575, "objective": "answer_ce+pair_contrastive", "optimizer_steps": 2, "options": {"batch_size": 2, "checkpoint_decoder": false, "device": "mps", "dtype": null, "epochs": 1, "gradient_accumulation_steps": 1, "limit": 8, "loss_chunk_size": 128, "lr": 0.0003, "max_grad_norm": 1.0, "max_steps": 2, "pair_contrastive_weight": 16.0, "shuffle": false, "supervise_eos": true, "validation_limit": 4, "weight_decay": 0.01}, "pair_contrastive_weight": 16.0, "seed": 42, "started_at": "2026-09-27T08:56:33.826000+00:00", "steps": 2, "supervise_eos": true, "window_size": 256}`

## Command

```sh
.venv/bin/python main.py train --data .datasets_cache/session-broad-pairs-256-v1 --output checkpoints/session-broad-all24-scaled-smoke-v1 --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 --memory-hidden-size 256 --memory-chunk-size 16 --memory-qk-scale 5.656854249492381 --aligned-qk-init --memory-gate-init -5.3 --window-size 256 --bptt-windows 4 --batch-size 2 --no-shuffle --pair-contrastive-weight 16 --supervise-eos --lr 0.0003 --device mps --no-checkpoint-decoder --validation-limit 4 --limit 8 --max-steps 2 --log-every 1
```
