# session-broad-full11-scratch-v1

Fresh layer-11 full-attention control matched to the linear-10 and shared-24 700-step runs.

Result: Familiar validation 117/160, 55/80 pairs; test 117/160, 49/80 pairs. Novel-value test 0/160. Saved-state validation 31/40.

- Checkpoint: `checkpoints/session-broad-full11-scratch-v1`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-broad-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_delta_read": false, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-27T08:00:22.698000+00:00", "episodes": 1400, "interrupted": false, "last_validation_loss": 0.4415268042148688, "loss_first_50": 4.408721470538481, "loss_last_50": 0.31989737138717783, "objective": "answer_ce+pair_contrastive", "optimizer_steps": 700, "options": {"batch_size": 2, "checkpoint_decoder": false, "device": "mps", "dtype": null, "epochs": 1, "gradient_accumulation_steps": 1, "limit": null, "loss_chunk_size": 128, "lr": 0.0003, "max_grad_norm": 1.0, "max_steps": null, "pair_contrastive_weight": 16.0, "shuffle": false, "supervise_eos": true, "validation_limit": 40, "weight_decay": 0.01}, "pair_contrastive_weight": 16.0, "seed": 42, "started_at": "2026-09-27T07:49:49.482000+00:00", "steps": 700, "supervise_eos": true, "window_size": 256}`

## Command

```sh
.venv/bin/python main.py train --data .datasets_cache/session-broad-pairs-256-v1 --output checkpoints/session-broad-full11-scratch-v1 --layers 11 --memory-hidden-size 256 --memory-chunk-size 16 --memory-qk-scale 5.656854249492381 --aligned-qk-init --window-size 256 --bptt-windows 4 --batch-size 2 --no-shuffle --pair-contrastive-weight 16 --supervise-eos --lr 0.0003 --device mps --no-checkpoint-decoder --validation-limit 40 --log-every 50
```

## Evaluations

- `broad-validation-all.json` (validation; full_prompt; data `.datasets_cache/session-broad-pairs-256-v1`):
  - normal: EM 0.73125, F1 0.7395833333333333, both pairs 55/80
  - By task: age 0/22, age_update 0/10, current_project 6/6, editor 6/6, name 4/10, preference 8/8, repo_build_tool 20/20, repo_config_file 10/14, repo_entry 10/10, repo_runtime 12/12, repo_test_command 12/12, repo_test_command_update 7/8, repo_update 10/10, timezone 12/12
- `novel-values-validation-all.json` (validation; full_prompt; data `.datasets_cache/session-broad-novel-values-v1`):
  - normal: EM 0.0, F1 0.125, both pairs 0/80
  - By task: age 0/22, age_update 0/10, current_project 0/6, editor 0/6, name 0/10, preference 0/8, repo_build_tool 0/20, repo_config_file 0/14, repo_entry 0/10, repo_runtime 0/12, repo_test_command 0/12, repo_test_command_update 0/8, repo_update 0/10, timezone 0/12
- `familiar-test-160.json` (test; full_prompt; data `.datasets_cache/session-broad-pairs-256-v1`):
  - normal: EM 0.73125, F1 0.7630208333333333, both pairs 49/80
  - By task: age 0/12, age_update 0/10, current_project 2/2, editor 12/12, name 6/14, preference 10/10, repo_build_tool 8/8, repo_config_file 14/18, repo_entry 11/16, repo_runtime 20/20, repo_test_command 7/8, repo_test_command_update 3/6, repo_update 12/12, timezone 12/12
- `novel-values-test-160.json` (test; full_prompt; data `.datasets_cache/session-broad-novel-values-v1`):
  - normal: EM 0.0, F1 0.15833333333333333, both pairs 0/80
  - By task: age 0/12, age_update 0/10, current_project 0/2, editor 0/12, name 0/14, preference 0/10, repo_build_tool 0/8, repo_config_file 0/18, repo_entry 0/16, repo_runtime 0/20, repo_test_command 0/8, repo_test_command_update 0/6, repo_update 0/12, timezone 0/12
- `state-validation-40.json` (validation; serialized_fast_state; data `.datasets_cache/session-broad-pairs-256-v1`):
  - state_only: EM 0.775, F1 0.7916666666666667, both pairs 14/20
  - reset: EM 0.05, F1 0.1125, both pairs 0/20
  - By task: age 0/2, current_project 2/2, editor 2/2, name 2/6, preference 2/2, repo_build_tool 4/4, repo_config_file 4/6, repo_entry 2/2, repo_runtime 4/4, repo_test_command_update 5/6, timezone 4/4
