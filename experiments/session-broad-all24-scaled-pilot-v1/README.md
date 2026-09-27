# session-broad-all24-scaled-pilot-v1

All 24 layers, initial gate logit -5.3; 200-step matched placement pilot.

Result: Familiar validation 6/40, 0/20 complete pairs; disabled 0/40, reset 2/40. Novel values 0/40. Saved-state 2/20 versus reset 0/20. Matched one-layer control 13/40 familiar, 5/20 saved-state.

- Checkpoint: `checkpoints/session-broad-all24-scaled-pilot-v1`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-broad-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_delta_read": false, "memory_gate_init": -5.3, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-27T14:48:57.220000+00:00", "episodes": 400, "interrupted": false, "last_validation_loss": null, "loss_first_50": 4.1128740310668945, "loss_last_50": 3.5106353577534866, "objective": "answer_ce+pair_contrastive", "optimizer_steps": 200, "options": {"batch_size": 2, "checkpoint_decoder": false, "device": "mps", "dtype": null, "epochs": 1, "gradient_accumulation_steps": 1, "limit": 400, "loss_chunk_size": 128, "lr": 0.0003, "max_grad_norm": 1.0, "max_steps": 200, "pair_contrastive_weight": 16.0, "shuffle": false, "supervise_eos": true, "validation_limit": 20, "weight_decay": 0.01}, "pair_contrastive_weight": 16.0, "seed": 42, "started_at": "2026-09-27T14:28:58.533000+00:00", "steps": 200, "supervise_eos": true, "window_size": 256}`

## Command

```sh
.venv/bin/python main.py train --data .datasets_cache/session-broad-pairs-256-v1 --output checkpoints/session-broad-all24-scaled-pilot-v1 --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 --memory-hidden-size 256 --memory-chunk-size 16 --memory-qk-scale 5.656854249492381 --aligned-qk-init --memory-gate-init -5.3 --window-size 256 --bptt-windows 4 --batch-size 2 --no-shuffle --pair-contrastive-weight 16 --supervise-eos --lr 0.0003 --device mps --no-checkpoint-decoder --validation-limit 20 --limit 400 --max-steps 200 --log-every 25
```

## Evaluations

- `familiar-validation-40.json` (validation; full_prompt; data `.datasets_cache/session-broad-pairs-256-v1`):
  - normal: EM 0.15, F1 0.18875, both pairs 0/20
  - disabled: EM 0.0, F1 0.0, both pairs 0/20
  - By task: age 0/2, current_project 0/2, editor 0/2, name 0/6, preference 1/2, repo_build_tool 1/4, repo_config_file 0/6, repo_entry 1/2, repo_runtime 0/4, repo_test_command_update 1/6, timezone 2/4
- `novel-values-validation-40.json` (validation; full_prompt; data `.datasets_cache/session-broad-novel-values-v1`):
  - normal: EM 0.0, F1 0.0675, both pairs 0/20
  - By task: age 0/2, current_project 0/2, editor 0/2, name 0/6, preference 0/2, repo_build_tool 0/4, repo_config_file 0/6, repo_entry 0/2, repo_runtime 0/4, repo_test_command_update 0/6, timezone 0/4
- `familiar-reset-40.json` (validation; full_prompt; data `.datasets_cache/session-broad-pairs-256-v1`):
  - reset: EM 0.05, F1 0.09740259740259741, both pairs 0/20
- `state-validation-20.json` (validation; serialized_fast_state; data `.datasets_cache/session-broad-pairs-256-v1`):
  - state_only: EM 0.1, F1 0.18571428571428572, both pairs 0/10
  - reset: EM 0.0, F1 0.0, both pairs 0/10
  - By task: age 0/2, current_project 0/2, name 0/2, preference 0/2, repo_build_tool 1/2, repo_config_file 0/2, repo_entry 0/2, repo_runtime 0/2, repo_test_command_update 0/2, timezone 1/2
