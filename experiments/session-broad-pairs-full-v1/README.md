# session-broad-pairs-full-v1

Interrupted after 154 steps: 150/700 pairs had unequal prompt token lengths, while the old ranking loss compared equal absolute positions. This checkpoint is diagnostic only.

Result: First 40 broad validation examples: 23/40 exact and 8/20 complete pairs; model trained with the misaligned ranking objective.

- Checkpoint: `checkpoints/session-broad-pairs-full-v1`
- Parent checkpoint: `checkpoints/session-pairs-contrastive-full-v1`
- Source data: `.datasets_cache/session-broad-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_delta_read": false, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T13:46:53.449000+00:00", "episodes": 1400, "interrupted": true, "last_validation_loss": null, "loss_first_50": 3.0776631432502057, "loss_last_50": 0.8821010487861459, "objective": "answer_ce+pair_contrastive", "optimizer_steps": 154, "options": {"batch_size": 2, "checkpoint_decoder": false, "device": "mps", "dtype": null, "epochs": 1, "gradient_accumulation_steps": 1, "limit": null, "loss_chunk_size": 128, "lr": 0.0003, "max_grad_norm": 1.0, "max_steps": null, "pair_contrastive_weight": 16.0, "shuffle": false, "supervise_eos": true, "validation_limit": 40, "weight_decay": 0.01}, "pair_contrastive_weight": 16.0, "seed": 42, "started_at": "2026-09-26T13:44:27.120000+00:00", "steps": 154, "supervise_eos": true, "window_size": 256}`

## Command

```sh
.venv/bin/python main.py train --data .datasets_cache/session-broad-pairs-256-v1 --checkpoint checkpoints/session-pairs-contrastive-full-v1 --output checkpoints/session-broad-pairs-full-v1 --window-size 256 --bptt-windows 4 --batch-size 2 --no-shuffle --pair-contrastive-weight 16 --supervise-eos --lr 0.0003 --device mps --no-checkpoint-decoder --validation-limit 40 --log-every 50
```

## Evaluations

- `validation-40.json` (validation):
  - normal: EM 0.575, F1 0.63, both pairs 8/20
