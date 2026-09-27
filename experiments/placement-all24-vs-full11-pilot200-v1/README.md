# All 24 layers versus one full-attention layer, 200-step pilot

Qwen3.5-0.8B has 24 decoder layers: 18 Gated DeltaNet and 6 full-attention.
This experiment installs a separate Titans adapter at every layer and compares
it with one adapter at full-attention layer 11. Both runs used the same seed,
first 400 paired training episodes, first 200 optimizer steps, answer plus
pair-ranking objective, window size 256, MPS, and all other training settings
from the preceding placement experiment. The all-layer run used an initial
memory-gate logit of `-5.3` per branch; the one-layer control used `-2.0`.
This scaling prevented the severe initial loss seen with 24 unscaled branches:
the first two losses were 13.39 and 49.77 unscaled, versus 7.39 and 6.37 scaled.
The unscaled two-step run is only a startup diagnostic.

| 200-step result | All 24 layers | Layer 11 |
| --- | ---: | ---: |
| Familiar validation exact | 6/40 | 13/40 |
| Familiar complete pairs | 0/20 | 2/20 |
| Memory disabled exact | 0/40 | 0/40 |
| Memory reset each window exact | 2/40 | 5/40 |
| Novel-value validation exact | 0/40 | 0/40 |
| Saved/reloaded state exact | 2/20 | 5/20 |
| Saved-state reset exact | 0/20 | 0/20 |
| Training loss, steps 151–200 | 3.680 | 2.495 |
| Step time, steps 1–200 | 1,191 s | 176 s |
| Adapter weights | 353 MB | 14.7 MB |

At this fixed 200-step budget, adapters on every layer were slower, much
larger, and less accurate on familiar examples. Both models were poor on
novel values. The all-layer run is a pilot, not proof that this placement
cannot learn with more compute or a different gate schedule. It also has
24 independent fast memories rather than one shared state read by every
layer. Cross-session persistence was tested by serializing and reloading
each episode's fast state before asking its question.

Artifacts:

- All-layer checkpoint: `checkpoints/session-broad-all24-scaled-pilot-v1`
- Single-layer control: `checkpoints/session-broad-full11-pilot200-v1`
- Unscaled startup diagnostic: `checkpoints/session-broad-all24-smoke-v1`
- Scaled startup diagnostic: `checkpoints/session-broad-all24-scaled-smoke-v1`
- Compact machine-readable comparison: `summary.json`

Each checkpoint folder contains its training log and full per-example
validation reports. Its `experiments/` record preserves the exact train
command and dataset manifest.
