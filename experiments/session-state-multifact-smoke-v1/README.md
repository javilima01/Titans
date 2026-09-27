# session-state-multifact-smoke-v1

Five separate chat processes shared one serialized fast state for two repositories and a later correction.

Result: Before correction: alpha/Bazel correct. After correction: alpha/Ninja was answered Bazel; beta/Meson was also answered Bazel. Scored 1/3 questions; state file 4.0 MB.

- Checkpoint: `checkpoints/session-broad-pairs-aligned-v1`
- Parent checkpoint: `checkpoints/session-pairs-contrastive-full-v1`
- Source data: `.datasets_cache/session-broad-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_delta_read": false, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-27T00:11:05.837000+00:00", "episodes": 1400, "interrupted": false, "last_validation_loss": 0.09348436502309945, "loss_first_50": 3.227038603451356, "loss_last_50": 0.15495721935673293, "objective": "answer_ce+pair_contrastive", "optimizer_steps": 700, "options": {"batch_size": 2, "checkpoint_decoder": false, "device": "mps", "dtype": null, "epochs": 1, "gradient_accumulation_steps": 1, "limit": null, "loss_chunk_size": 128, "lr": 0.0003, "max_grad_norm": 1.0, "max_steps": null, "pair_contrastive_weight": 16.0, "shuffle": false, "supervise_eos": true, "validation_limit": 40, "weight_decay": 0.01}, "pair_contrastive_weight": 16.0, "seed": 42, "started_at": "2026-09-27T00:00:23.114000+00:00", "steps": 700, "supervise_eos": true, "window_size": 256}`

## Command

```sh
main.py chat --checkpoint checkpoints/session-broad-pairs-aligned-v1 --device mps --window-size 256 --memory-state-file /private/tmp/codex-memory-multifact-20260927.safetensors --system 'Answer questions with only the requested value, without explanation.' --prompt '<one prompt from results.json>' --max-new-tokens 12
# Run one process per prompt, in the order shown in results.json.
```

Other reports:

- `experiments/session-state-multifact-smoke-v1/results.json`
