# session-broad-pairs-aligned-v1

Strong single-fact recall on trained synthetic templates; novel values, unseen categories, and multi-fact corrections remain unreliable.

Result: Broad held-out test 138/160 and 67/80 pairs, disabled 0/160, reset 12/160. Novel-value matched test neural 2/160 and 0/80 pairs; lexical retrieval 150/160 and 71/80 pairs with 160/160 support hits. Serialized state single-fact check 38/40; five-process two-fact correction smoke 1/3 scored questions. Unseen categories 12/160.

- Checkpoint: `checkpoints/session-broad-pairs-aligned-v1`
- Parent checkpoint: `checkpoints/session-pairs-contrastive-full-v1`
- Source data: `.datasets_cache/session-broad-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_delta_read": false, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-27T00:11:05.837000+00:00", "episodes": 1400, "interrupted": false, "last_validation_loss": 0.09348436502309945, "loss_first_50": 3.227038603451356, "loss_last_50": 0.15495721935673293, "objective": "answer_ce+pair_contrastive", "optimizer_steps": 700, "options": {"batch_size": 2, "checkpoint_decoder": false, "device": "mps", "dtype": null, "epochs": 1, "gradient_accumulation_steps": 1, "limit": null, "loss_chunk_size": 128, "lr": 0.0003, "max_grad_norm": 1.0, "max_steps": null, "pair_contrastive_weight": 16.0, "shuffle": false, "supervise_eos": true, "validation_limit": 40, "weight_decay": 0.01}, "pair_contrastive_weight": 16.0, "seed": 42, "started_at": "2026-09-27T00:00:23.114000+00:00", "steps": 700, "supervise_eos": true, "window_size": 256}`

## Command

```sh
.venv/bin/python main.py train --data .datasets_cache/session-broad-pairs-256-v1 --checkpoint checkpoints/session-pairs-contrastive-full-v1 --output checkpoints/session-broad-pairs-aligned-v1 --window-size 256 --bptt-windows 4 --batch-size 2 --no-shuffle --pair-contrastive-weight 16 --supervise-eos --lr 0.0003 --device mps --no-checkpoint-decoder --validation-limit 40 --log-every 50
```

## Evaluations

- `broad-validation-all.json` (validation; full_prompt; data `.datasets_cache/session-broad-pairs-256-v1`):
  - normal: EM 0.8125, F1 0.8208333333333334, both pairs 62/80
  - disabled: EM 0.0, F1 0.0, both pairs 0/80
  - reset: EM 0.06875, F1 0.17098214285714286, both pairs 0/80
  - By task: age 4/22, age_update 0/10, current_project 6/6, editor 6/6, name 10/10, preference 8/8, repo_build_tool 20/20, repo_config_file 12/14, repo_entry 10/10, repo_runtime 12/12, repo_test_command 12/12, repo_test_command_update 8/8, repo_update 10/10, timezone 12/12
- `unseen-categories-validation.json` (validation; full_prompt; data `.datasets_cache/session-unseen-pairs-256-v1`):
  - normal: EM 0.075, F1 0.094375, both pairs 0/80
  - By task: repo_deploy_region 11/60, repo_release_branch 1/42, user_work_hours 0/58
- `novel-values-validation.json` (validation; full_prompt; data `.datasets_cache/session-broad-novel-values-v1`):
  - normal: EM 0.0, F1 0.16145833333333331, both pairs 0/80
  - By task: age 0/22, age_update 0/10, current_project 0/6, editor 0/6, name 0/10, preference 0/8, repo_build_tool 0/20, repo_config_file 0/14, repo_entry 0/10, repo_runtime 0/12, repo_test_command 0/12, repo_test_command_update 0/8, repo_update 0/10, timezone 0/12
- `novel-values-full-context-40.json` (validation; full_prompt; data `.datasets_cache/session-broad-novel-values-v1`):
  - disabled: EM 0.95, F1 0.95, both pairs 18/20
- `serialized-state-validation-40.json` (validation; serialized_fast_state; data `.datasets_cache/session-broad-pairs-256-v1`):
  - state_only: EM 0.95, F1 0.9666666666666666, both pairs 18/20
  - reset: EM 0.05, F1 0.0875, both pairs 0/20
  - By task: age 1/2, current_project 2/2, editor 2/2, name 6/6, preference 2/2, repo_build_tool 4/4, repo_config_file 5/6, repo_entry 2/2, repo_runtime 4/4, repo_test_command_update 6/6, timezone 4/4
- `broad-test-all.json` (test; full_prompt; data `.datasets_cache/session-broad-pairs-256-v1`):
  - normal: EM 0.8625, F1 0.875, both pairs 67/80
  - disabled: EM 0.0, F1 0.0, both pairs 0/80
  - reset: EM 0.075, F1 0.19821428571428573, both pairs 0/80
  - By task: age 3/12, age_update 0/10, current_project 2/2, editor 12/12, name 14/14, preference 10/10, repo_build_tool 8/8, repo_config_file 15/18, repo_entry 16/16, repo_runtime 20/20, repo_test_command 8/8, repo_test_command_update 6/6, repo_update 12/12, timezone 12/12
- `retrieval-novel-values-40.json` (validation; text_retrieval_control; data `.datasets_cache/session-broad-novel-values-v1`):
  - lexical: EM 0.925, F1 0.9321428571428572, both pairs 18/20
  - oracle: EM 0.925, F1 0.9321428571428572, both pairs 18/20
- `retrieval-novel-values-test-160.json` (test; text_retrieval_control; data `.datasets_cache/session-broad-novel-values-v1`):
  - lexical: EM 0.9375, F1 0.9491071428571429, both pairs 71/80
- `novel-values-test-all.json` (test; full_prompt; data `.datasets_cache/session-broad-novel-values-v1`):
  - normal: EM 0.0125, F1 0.1895089285714286, both pairs 0/80
  - By task: age 0/12, age_update 0/10, current_project 0/2, editor 0/12, name 0/14, preference 0/10, repo_build_tool 0/8, repo_config_file 0/18, repo_entry 0/16, repo_runtime 1/20, repo_test_command 0/8, repo_test_command_update 1/6, repo_update 0/12, timezone 0/12
