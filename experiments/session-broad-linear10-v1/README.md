# session-broad-linear10-v1

Titans branch on Qwen linear Gated DeltaNet layer 10; matched 700-step placement ablation.

Result: Familiar validation 123/160, 59/80 pairs; disabled 0/160, reset 13/160. Novel-value validation 1/160, 0/80 pairs. Matched fresh full layer 11: 117/160 familiar, 0/160 novel.

- Checkpoint: `checkpoints/session-broad-linear10-v1`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-broad-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [10], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_delta_read": false, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-27T07:49:33.278000+00:00", "episodes": 1400, "interrupted": false, "last_validation_loss": 0.20531605451534957, "loss_first_50": 4.366112994559018, "loss_last_50": 0.3271769585990114, "objective": "answer_ce+pair_contrastive", "optimizer_steps": 700, "options": {"batch_size": 2, "checkpoint_decoder": false, "device": "mps", "dtype": null, "epochs": 1, "gradient_accumulation_steps": 1, "limit": null, "loss_chunk_size": 128, "lr": 0.0003, "max_grad_norm": 1.0, "max_steps": null, "pair_contrastive_weight": 16.0, "shuffle": false, "supervise_eos": true, "validation_limit": 40, "weight_decay": 0.01}, "pair_contrastive_weight": 16.0, "seed": 42, "started_at": "2026-09-27T07:38:49.478000+00:00", "steps": 700, "supervise_eos": true, "window_size": 256}`

## Command

```sh
.venv/bin/python main.py train --data .datasets_cache/session-broad-pairs-256-v1 --output checkpoints/session-broad-linear10-v1 --layers 10 --memory-hidden-size 256 --memory-chunk-size 16 --memory-qk-scale 5.656854249492381 --aligned-qk-init --window-size 256 --bptt-windows 4 --batch-size 2 --no-shuffle --pair-contrastive-weight 16 --supervise-eos --lr 0.0003 --device mps --no-checkpoint-decoder --validation-limit 40 --log-every 50
```

## Evaluations

- `broad-validation-all.json` (validation; full_prompt; data `.datasets_cache/session-broad-pairs-256-v1`):
  - normal: EM 0.76875, F1 0.7729166666666667, both pairs 59/80
  - disabled: EM 0.0, F1 0.0, both pairs 0/80
  - reset: EM 0.08125, F1 0.18095238095238095, both pairs 0/80
  - By task: age 1/22, age_update 0/10, current_project 6/6, editor 6/6, name 6/10, preference 8/8, repo_build_tool 20/20, repo_config_file 12/14, repo_entry 10/10, repo_runtime 12/12, repo_test_command 12/12, repo_test_command_update 8/8, repo_update 10/10, timezone 12/12
- `novel-values-validation-all.json` (validation; full_prompt; data `.datasets_cache/session-broad-novel-values-v1`):
  - normal: EM 0.00625, F1 0.19702380952380952, both pairs 0/80
  - By task: age 0/22, age_update 0/10, current_project 0/6, editor 0/6, name 0/10, preference 0/8, repo_build_tool 1/20, repo_config_file 0/14, repo_entry 0/10, repo_runtime 0/12, repo_test_command 0/12, repo_test_command_update 0/8, repo_update 0/10, timezone 0/12
