# session-broad-full11-pilot200-v1

Fresh full-attention layer 11 control for the all-24-layer 200-step pilot.

Result: Familiar validation 13/40, 2/20 complete pairs; disabled 0/40, reset 5/40. Novel values 0/40. Saved-state 5/20 versus reset 0/20.

- Checkpoint: `checkpoints/session-broad-full11-pilot200-v1`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-broad-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_delta_read": false, "memory_gate_init": null, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-27T14:52:10.301000+00:00", "episodes": 400, "interrupted": false, "last_validation_loss": null, "loss_first_50": 4.408721470538481, "loss_last_50": 2.4165473925839565, "objective": "answer_ce+pair_contrastive", "optimizer_steps": 200, "options": {"batch_size": 2, "checkpoint_decoder": false, "device": "mps", "dtype": null, "epochs": 1, "gradient_accumulation_steps": 1, "limit": 400, "loss_chunk_size": 128, "lr": 0.0003, "max_grad_norm": 1.0, "max_steps": 200, "pair_contrastive_weight": 16.0, "shuffle": false, "supervise_eos": true, "validation_limit": 20, "weight_decay": 0.01}, "pair_contrastive_weight": 16.0, "seed": 42, "started_at": "2026-09-27T14:49:13.036000+00:00", "steps": 200, "supervise_eos": true, "window_size": 256}`

## Command

```sh
.venv/bin/python main.py train --data .datasets_cache/session-broad-pairs-256-v1 --output checkpoints/session-broad-full11-pilot200-v1 --layers 11 --memory-hidden-size 256 --memory-chunk-size 16 --memory-qk-scale 5.656854249492381 --aligned-qk-init --window-size 256 --bptt-windows 4 --batch-size 2 --no-shuffle --pair-contrastive-weight 16 --supervise-eos --lr 0.0003 --device mps --no-checkpoint-decoder --validation-limit 20 --limit 400 --max-steps 200 --log-every 25
```

## Evaluations

- `familiar-validation-40.json` (validation; full_prompt; data `.datasets_cache/session-broad-pairs-256-v1`):
  - normal: EM 0.325, F1 0.37041666666666667, both pairs 2/20
  - disabled: EM 0.0, F1 0.0, both pairs 0/20
  - By task: age 0/2, current_project 0/2, editor 1/2, name 1/6, preference 1/2, repo_build_tool 3/4, repo_config_file 2/6, repo_entry 0/2, repo_runtime 1/4, repo_test_command_update 3/6, timezone 1/4
- `novel-values-validation-40.json` (validation; full_prompt; data `.datasets_cache/session-broad-novel-values-v1`):
  - normal: EM 0.0, F1 0.07458333333333333, both pairs 0/20
  - By task: age 0/2, current_project 0/2, editor 0/2, name 0/6, preference 0/2, repo_build_tool 0/4, repo_config_file 0/6, repo_entry 0/2, repo_runtime 0/4, repo_test_command_update 0/6, timezone 0/4
- `familiar-reset-40.json` (validation; full_prompt; data `.datasets_cache/session-broad-pairs-256-v1`):
  - reset: EM 0.125, F1 0.20089285714285715, both pairs 0/20
- `state-validation-20.json` (validation; serialized_fast_state; data `.datasets_cache/session-broad-pairs-256-v1`):
  - state_only: EM 0.25, F1 0.35, both pairs 0/10
  - reset: EM 0.0, F1 0.04285714285714286, both pairs 0/10
  - By task: age 0/2, current_project 0/2, name 0/2, preference 1/2, repo_build_tool 1/2, repo_config_file 1/2, repo_entry 0/2, repo_runtime 0/2, repo_test_command_update 1/2, timezone 1/2
