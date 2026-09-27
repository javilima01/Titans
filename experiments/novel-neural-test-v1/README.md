# novel-neural-test-v1

Matched 160-example novel-value test for trained neural memory; all answer values excluded from training pools.

Result: Normal 2/160 exact, 0/80 complete pairs; lexical retrieval on the same test items answered 150/160 and 71/80 pairs.

- Checkpoint: `checkpoints/session-broad-pairs-aligned-v1`
- Parent checkpoint: `checkpoints/session-pairs-contrastive-full-v1`
- Source data: `.datasets_cache/session-broad-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_delta_read": false, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-27T00:11:05.837000+00:00", "episodes": 1400, "interrupted": false, "last_validation_loss": 0.09348436502309945, "loss_first_50": 3.227038603451356, "loss_last_50": 0.15495721935673293, "objective": "answer_ce+pair_contrastive", "optimizer_steps": 700, "options": {"batch_size": 2, "checkpoint_decoder": false, "device": "mps", "dtype": null, "epochs": 1, "gradient_accumulation_steps": 1, "limit": null, "loss_chunk_size": 128, "lr": 0.0003, "max_grad_norm": 1.0, "max_steps": null, "pair_contrastive_weight": 16.0, "shuffle": false, "supervise_eos": true, "validation_limit": 40, "weight_decay": 0.01}, "pair_contrastive_weight": 16.0, "seed": 42, "started_at": "2026-09-27T00:00:23.114000+00:00", "steps": 700, "supervise_eos": true, "window_size": 256}`

## Command

```sh
.venv/bin/python main.py test --data .datasets_cache/session-broad-novel-values-v1 --checkpoint checkpoints/session-broad-pairs-aligned-v1 --device mps --window-size 256 --max-new-tokens 12 --memory-modes normal --report checkpoints/session-broad-pairs-aligned-v1/novel-values-test-all.json
```

## Evaluations

- `novel-values-test-all.json` (test; full_prompt; data `.datasets_cache/session-broad-novel-values-v1`):
  - normal: EM 0.0125, F1 0.1895089285714286, both pairs 0/80
  - By task: age 0/12, age_update 0/10, current_project 0/2, editor 0/12, name 0/14, preference 0/10, repo_build_tool 0/8, repo_config_file 0/18, repo_entry 0/16, repo_runtime 1/20, repo_test_command 0/8, repo_test_command_update 1/6, repo_update 0/12, timezone 0/12
