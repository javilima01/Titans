# session-age-pairs-pilot-v1

Age-focused pilot with all differing answer tokens and EOS supervision: changed answers, but zero fully correct pairs.

Result: Age-only pilot validation: 1/24 exact answers, zero fully correct pairs; 9/12 pairs changed their predictions.

- Checkpoint: `checkpoints/session-age-pairs-pilot-v1`
- Parent checkpoint: `checkpoints/session-pairs-contrastive-full-v1`
- Source data: `.datasets_cache/session-age-pairs-256-v1`
- Adapter: `{"aligned_qk_init": true, "checkpoint_memory": true, "layer_indices": [11], "max_inner_grad_norm": 1.0, "memory_chunk_size": 16, "memory_delta_read": false, "memory_hidden_size": 256, "memory_qk_scale": 5.656854249492381}`
- Training: `{"bptt_windows": 4, "ended_at": "2026-09-26T13:25:58.721000+00:00", "episodes": 100, "interrupted": false, "last_validation_loss": 1.6545411745707195, "loss_first_50": 2.947459789911906, "loss_last_50": 2.947459789911906, "objective": "answer_ce+pair_contrastive", "optimizer_steps": 50, "options": null, "pair_contrastive_weight": 16.0, "seed": 42, "started_at": "2026-09-26T13:25:01.512000+00:00", "steps": 50, "supervise_eos": true, "window_size": 256}`

## Evaluations

- `validation-24.json` (validation):
  - normal: EM 0.041666666666666664, F1 0.041666666666666664, both pairs 0/12
  - disabled: EM 0.041666666666666664, F1 0.041666666666666664, both pairs 0/12
  - reset: EM 0.0, F1 0.0, both pairs 0/12
