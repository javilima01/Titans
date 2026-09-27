# session-age-pairs-full-v1

Interrupted after 249 steps when a teacher-forcing shortcut was found in the later-digit ranking loss; partial age checkpoint retained for diagnosis.

Result: Age-only validation (24 examples): normal 6/24 exact, 1/12 pairs fully correct; disabled 1/24, reset 1/24.

- Checkpoint: `checkpoints/session-age-pairs-full-v1`
- Parent checkpoint: `checkpoints/session-pairs-contrastive-full-v1`
- Source data: `.datasets_cache/session-age-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_delta_read": false, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T13:38:05.407000+00:00", "episodes": 1200, "interrupted": true, "last_validation_loss": null, "loss_first_50": 2.9473253758748372, "loss_last_50": 1.2873398892084758, "objective": "answer_ce+pair_contrastive", "optimizer_steps": 249, "options": {"batch_size": 2, "checkpoint_decoder": false, "device": "mps", "dtype": null, "epochs": 1, "gradient_accumulation_steps": 1, "limit": null, "loss_chunk_size": 128, "lr": 0.0003, "max_grad_norm": 1.0, "max_steps": null, "pair_contrastive_weight": 16.0, "shuffle": false, "supervise_eos": true, "validation_limit": 24, "weight_decay": 0.01}, "pair_contrastive_weight": 16.0, "seed": 42, "started_at": "2026-09-26T13:33:44.997000+00:00", "steps": 249, "supervise_eos": true, "window_size": 256}`

## Command

```sh
.venv/bin/python main.py train --data .datasets_cache/session-age-pairs-256-v1 --checkpoint checkpoints/session-pairs-contrastive-full-v1 --output checkpoints/session-age-pairs-full-v1 --window-size 256 --bptt-windows 4 --batch-size 2 --no-shuffle --pair-contrastive-weight 16 --supervise-eos --lr 0.0003 --device mps --no-checkpoint-decoder --validation-limit 24 --log-every 50
```

## Evaluations

- `validation-24.json` (validation):
  - normal: EM 0.25, F1 0.25, both pairs 1/12
  - disabled: EM 0.041666666666666664, F1 0.041666666666666664, both pairs 0/12
  - reset: EM 0.041666666666666664, F1 0.041666666666666664, both pairs 0/12
