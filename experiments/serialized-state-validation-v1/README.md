# serialized-state-validation-v1

Controlled cross-session test: save and reload fast memory after prior sessions; ask with only the new-session prompt.

Result: Serialized state: 38/40 exact and 18/20 pairs; reset: 2/40 exact and 0/20 pairs.

- Checkpoint: `checkpoints/session-broad-pairs-aligned-v1`
- Parent checkpoint: `checkpoints/session-pairs-contrastive-full-v1`
- Source data: `.datasets_cache/session-broad-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_delta_read": false, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-27T00:11:05.837000+00:00", "episodes": 1400, "interrupted": false, "last_validation_loss": 0.09348436502309945, "loss_first_50": 3.227038603451356, "loss_last_50": 0.15495721935673293, "objective": "answer_ce+pair_contrastive", "optimizer_steps": 700, "options": {"batch_size": 2, "checkpoint_decoder": false, "device": "mps", "dtype": null, "epochs": 1, "gradient_accumulation_steps": 1, "limit": null, "loss_chunk_size": 128, "lr": 0.0003, "max_grad_norm": 1.0, "max_steps": null, "pair_contrastive_weight": 16.0, "shuffle": false, "supervise_eos": true, "validation_limit": 40, "weight_decay": 0.01}, "pair_contrastive_weight": 16.0, "seed": 42, "started_at": "2026-09-27T00:00:23.114000+00:00", "steps": 700, "supervise_eos": true, "window_size": 256}`

## Command

```sh
MPLCONFIGDIR=/private/tmp/codex-mpl .venv/bin/python -m src.llm.helpers.state_evaluation --data .datasets_cache/session-broad-pairs-256-v1 --checkpoint checkpoints/session-broad-pairs-aligned-v1 --device mps --limit 40 --max-new-tokens 12 --report checkpoints/session-broad-pairs-aligned-v1/serialized-state-validation-40.json
```

## Evaluations

- `serialized-state-validation-40.json` (validation; serialized_fast_state; data `.datasets_cache/session-broad-pairs-256-v1`):
  - state_only: EM 0.95, F1 0.9666666666666666, both pairs 18/20
  - reset: EM 0.05, F1 0.0875, both pairs 0/20
  - By task: age 1/2, current_project 2/2, editor 2/2, name 6/6, preference 2/2, repo_build_tool 4/4, repo_config_file 5/6, repo_entry 2/2, repo_runtime 4/4, repo_test_command_update 6/6, timezone 4/4
