# session-pairs-contrastive-full-v1

Paired ranking objective yields a clear memory effect on held-out session facts.

Result: Validation: 56/120 exact answers; 24/60 pairs fully correct; reset 6/120 answers and 0 pairs; disabled 2/120 answers and 0 pairs.

- Checkpoint: `checkpoints/session-pairs-contrastive-full-v1`
- Parent checkpoint: `checkpoints/session-pairs-full-v1`
- Source data: `.datasets_cache/session-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_delta_read": false, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T13:17:26.829000+00:00", "episodes": 1200, "interrupted": false, "last_validation_loss": 0.5584971639845107, "loss_first_50": 3.8320869505405426, "loss_last_50": 1.4933272532838504, "objective": "answer_ce+pair_contrastive", "optimizer_steps": 600, "options": null, "pair_contrastive_weight": 16.0, "seed": 42, "started_at": "2026-09-26T13:07:59.981000+00:00", "steps": 600, "supervise_eos": null, "window_size": 256}`

## Evaluations

- `validation-24.json` (validation; full_prompt):
  - normal: EM 0.5833333333333334, F1 0.6809764309764309, both pairs 6/12
  - disabled: EM 0.0, F1 0.0, both pairs 0/12
  - reset: EM 0.041666666666666664, F1 0.4204545454545454, both pairs 0/12
  - By task: age 0/4, age_update 0/2, name 2/2, preference 0/2, repo_entry 8/10, repo_update 4/4
- `validation-all.json` (validation; full_prompt):
  - normal: EM 0.4666666666666667, F1 0.4954545454545454, both pairs 24/60
  - disabled: EM 0.016666666666666666, F1 0.016666666666666666, both pairs 0/60
  - reset: EM 0.05, F1 0.2639520202020202, both pairs 0/60
  - By task: age 0/28, age_update 0/26, name 11/14, preference 5/10, repo_entry 20/22, repo_update 20/20
- `broad-validation-before.json` (validation; full_prompt; data `.datasets_cache/session-broad-pairs-256-v1`):
  - normal: EM 0.16875, F1 0.20275523088023087, both pairs 10/80
  - By task: age 0/22, age_update 0/10, current_project 0/6, editor 0/6, name 7/10, preference 1/8, repo_build_tool 0/20, repo_config_file 0/14, repo_entry 8/10, repo_runtime 1/12, repo_test_command 0/12, repo_test_command_update 0/8, repo_update 10/10, timezone 0/12
