# session-profile-256-strong-v1

Longer cross-session profile run; early generation evaluation did not establish general recall.

- Checkpoint: `checkpoints/session-profile-256-strong-v1`
- Parent checkpoint: `none`
- Source data: `.datasets_cache/session-profile-256-v1`
- Adapter: `{"checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_hidden_size": 256}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T11:34:08.714000+00:00", "episodes": 1200, "interrupted": true, "last_validation_loss": null, "loss_first_50": 2.3796387075989567, "loss_last_50": 2.3796387075989567, "objective": "answer_ce", "optimizer_steps": 50, "options": null, "seed": 42, "started_at": "2026-09-26T11:16:13.352000+00:00", "steps": 50, "supervise_eos": null, "window_size": 256}`

## Evaluations

- `early-validation.json` (validation):
  - normal: EM 0.08333333333333333, F1 0.08333333333333333, both pairs n/a/n/a
  - disabled: EM 0.0, F1 0.0, both pairs n/a/n/a
  - reset: EM 0.0, F1 0.0, both pairs n/a/n/a
