# session-broad-shared24-full-v1

Full 700-step shared-state experiment: all 24 layers read one fast state, written once per Qwen window.

Result: Familiar test 124/160, 58/80 pairs; validation 121/160, 55/80. Novel-value test 2/160, 0/80 pairs. Saved-state validation 30/40; five-process multi-fact smoke 1/3.

- Checkpoint: `checkpoints/session-broad-shared24-full-v1`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-broad-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_delta_read": false, "memory_gate_init": -5.3, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381, "shared_across_layers": true}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-27T15:55:14.940000+00:00", "episodes": 1400, "interrupted": false, "last_validation_loss": 0.25236919598701674, "loss_first_50": 4.384763903088039, "loss_last_50": 0.17744067250318132, "objective": "answer_ce+pair_contrastive", "optimizer_steps": 700, "options": {"batch_size": 2, "checkpoint_decoder": false, "device": "mps", "dtype": null, "epochs": 1, "gradient_accumulation_steps": 1, "limit": null, "loss_chunk_size": 128, "lr": 0.0003, "max_grad_norm": 1.0, "max_steps": null, "pair_contrastive_weight": 16.0, "shuffle": false, "supervise_eos": true, "validation_limit": 40, "weight_decay": 0.01}, "pair_contrastive_weight": 16.0, "seed": 42, "started_at": "2026-09-27T15:43:02.619000+00:00", "steps": 700, "supervise_eos": true, "window_size": 256}`

## Command

```sh
.venv/bin/python main.py train --data .datasets_cache/session-broad-pairs-256-v1 --output checkpoints/session-broad-shared24-full-v1 --shared-memory --memory-hidden-size 256 --memory-chunk-size 16 --memory-qk-scale 5.656854249492381 --aligned-qk-init --window-size 256 --bptt-windows 4 --batch-size 2 --no-shuffle --pair-contrastive-weight 16 --supervise-eos --lr 0.0003 --device mps --no-checkpoint-decoder --validation-limit 40 --log-every 50
```

## Evaluations

- `familiar-validation-160.json` (validation; full_prompt; data `.datasets_cache/session-broad-pairs-256-v1`):
  - normal: EM 0.75625, F1 0.7893939393939393, both pairs 55/80
  - disabled: EM 0.0, F1 0.0, both pairs 0/80
  - reset: EM 0.0, F1 0.0, both pairs 0/80
  - By task: age 5/22, age_update 0/10, current_project 6/6, editor 6/6, name 8/10, preference 8/8, repo_build_tool 20/20, repo_config_file 5/14, repo_entry 10/10, repo_runtime 12/12, repo_test_command 12/12, repo_test_command_update 8/8, repo_update 9/10, timezone 12/12
- `novel-values-validation-160.json` (validation; full_prompt; data `.datasets_cache/session-broad-novel-values-v1`):
  - normal: EM 0.0125, F1 0.15197916666666667, both pairs 0/80
  - By task: age 0/22, age_update 0/10, current_project 0/6, editor 0/6, name 0/10, preference 1/8, repo_build_tool 0/20, repo_config_file 0/14, repo_entry 0/10, repo_runtime 0/12, repo_test_command 1/12, repo_test_command_update 0/8, repo_update 0/10, timezone 0/12
- `state-validation-40.json` (validation; serialized_fast_state; data `.datasets_cache/session-broad-pairs-256-v1`):
  - state_only: EM 0.75, F1 0.7833333333333333, both pairs 12/20
  - reset: EM 0.0, F1 0.0, both pairs 0/20
  - By task: age 0/2, current_project 1/2, editor 1/2, name 5/6, preference 2/2, repo_build_tool 4/4, repo_config_file 5/6, repo_entry 2/2, repo_runtime 2/4, repo_test_command_update 4/6, timezone 4/4
- `novel-values-test-160.json` (test; full_prompt; data `.datasets_cache/session-broad-novel-values-v1`):
  - normal: EM 0.0125, F1 0.1807730463980464, both pairs 0/80
  - By task: age 0/12, age_update 0/10, current_project 0/2, editor 0/12, name 0/14, preference 1/10, repo_build_tool 0/8, repo_config_file 0/18, repo_entry 0/16, repo_runtime 0/20, repo_test_command 0/8, repo_test_command_update 1/6, repo_update 0/12, timezone 0/12
- `familiar-test-160.json` (test; full_prompt; data `.datasets_cache/session-broad-pairs-256-v1`):
  - normal: EM 0.775, F1 0.8314285714285715, both pairs 58/80
  - By task: age 3/12, age_update 0/10, current_project 2/2, editor 12/12, name 12/14, preference 10/10, repo_build_tool 8/8, repo_config_file 4/18, repo_entry 16/16, repo_runtime 20/20, repo_test_command 8/8, repo_test_command_update 5/6, repo_update 12/12, timezone 12/12
