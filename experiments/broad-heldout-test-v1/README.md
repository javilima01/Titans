# broad-heldout-test-v1

Held-out 14-category test after checkpoint selection on validation.

Result: Normal 138/160, 67/80 pairs; disabled 0/160 and reset 12/160, each 0/80 pairs.

- Checkpoint: `checkpoints/session-broad-pairs-aligned-v1`
- Parent checkpoint: `checkpoints/session-pairs-contrastive-full-v1`
- Source data: `.datasets_cache/session-broad-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_delta_read": false, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-27T00:11:05.837000+00:00", "episodes": 1400, "interrupted": false, "last_validation_loss": 0.09348436502309945, "loss_first_50": 3.227038603451356, "loss_last_50": 0.15495721935673293, "objective": "answer_ce+pair_contrastive", "optimizer_steps": 700, "options": {"batch_size": 2, "checkpoint_decoder": false, "device": "mps", "dtype": null, "epochs": 1, "gradient_accumulation_steps": 1, "limit": null, "loss_chunk_size": 128, "lr": 0.0003, "max_grad_norm": 1.0, "max_steps": null, "pair_contrastive_weight": 16.0, "shuffle": false, "supervise_eos": true, "validation_limit": 40, "weight_decay": 0.01}, "pair_contrastive_weight": 16.0, "seed": 42, "started_at": "2026-09-27T00:00:23.114000+00:00", "steps": 700, "supervise_eos": true, "window_size": 256}`

## Command

```sh
.venv/bin/python main.py test --data .datasets_cache/session-broad-pairs-256-v1 --checkpoint checkpoints/session-broad-pairs-aligned-v1 --device mps --window-size 256 --max-new-tokens 12 --ablations --report checkpoints/session-broad-pairs-aligned-v1/broad-test-all.json
```

## Evaluations

- `broad-test-all.json` (test; full_prompt; data `.datasets_cache/session-broad-pairs-256-v1`):
  - normal: EM 0.8625, F1 0.875, both pairs 67/80
  - disabled: EM 0.0, F1 0.0, both pairs 0/80
  - reset: EM 0.075, F1 0.19821428571428573, both pairs 0/80
  - By task: age 3/12, age_update 0/10, current_project 2/2, editor 12/12, name 14/14, preference 10/10, repo_build_tool 8/8, repo_config_file 15/18, repo_entry 16/16, repo_runtime 20/20, repo_test_command 8/8, repo_test_command_update 6/6, repo_update 12/12, timezone 12/12
