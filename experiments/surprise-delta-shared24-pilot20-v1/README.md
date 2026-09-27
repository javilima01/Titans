# surprise-delta-shared24-pilot20-v1

- Checkpoint: `checkpoints/surprise-delta-shared24-pilot20-v1`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-broad-pairs-256-v1`
- Adapter: `{"checkpoint_memory": true, "layer_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], "memory_chunk_size": 16, "memory_gate_init": -5.3, "memory_hidden_size": 128, "memory_type": "surprise_delta", "shared_across_layers": true}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-27T16:35:49.953000+00:00", "episodes": 1400, "interrupted": false, "last_validation_loss": null, "loss_first_50": 5.315065998142049, "loss_last_50": 5.315065998142049, "objective": "answer_ce+pair_contrastive", "optimizer_steps": 20, "options": {"batch_size": 2, "checkpoint_decoder": false, "device": "mps", "dtype": null, "epochs": 1, "gradient_accumulation_steps": 1, "limit": null, "loss_chunk_size": 128, "lr": 0.0003, "max_grad_norm": 1.0, "max_steps": 20, "pair_contrastive_weight": 16.0, "shuffle": false, "supervise_eos": true, "validation_limit": 20, "weight_decay": 0.01}, "pair_contrastive_weight": 16.0, "seed": 42, "started_at": "2026-09-27T16:35:28.893000+00:00", "steps": 20, "supervise_eos": true, "window_size": 256}`

## Command

```sh
.venv/bin/python main.py train --data .datasets_cache/session-broad-pairs-256-v1 --output checkpoints/surprise-delta-shared24-pilot20-v1 --memory-type surprise_delta --shared-memory --memory-hidden-size 128 --memory-chunk-size 16 --window-size 256 --bptt-windows 4 --batch-size 2 --no-shuffle --pair-contrastive-weight 16 --supervise-eos --lr 0.0003 --device mps --no-checkpoint-decoder --validation-limit 20 --max-steps 20 --log-every 5
```

## Evaluations

- `familiar-validation-40.json` (validation; full_prompt; data `.datasets_cache/session-broad-pairs-256-v1`):
  - normal: EM 0.025, F1 0.075, both pairs 0/20
  - disabled: EM 0.0, F1 0.0, both pairs 0/20
  - By task: age 0/2, current_project 0/2, editor 0/2, name 0/6, preference 0/2, repo_build_tool 0/4, repo_config_file 0/6, repo_entry 0/2, repo_runtime 1/4, repo_test_command_update 0/6, timezone 0/4
