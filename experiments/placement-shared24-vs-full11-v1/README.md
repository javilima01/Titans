# One shared memory across 24 layers

This experiment tests one Titans module and one fast state read by all 24
Qwen3.5-0.8B decoder layers (18 Gated DeltaNet, 6 full attention). Every layer
reads the state snapshot from before the current Qwen window. A learned
weighted mixture of all layer inputs writes the next state once after the
window. Reads subtract the initial memory MLP, leaving the first window's
frozen backbone unchanged. This ordering prevents later tokens from changing
earlier reads within a window.

The write weights are learned but fixed across tokens, rather than computed
from each token's content. After training, their softmax ranged from 0.0389
to 0.0453 (uniform is 0.0417), so the model used a near-uniform mixture.
Content-dependent cross-layer attention remains a separate architecture test.

The shared run and fresh layer-11 control each trained from seed 42 for 700
steps on the same 1,400 paired episodes, with a 256-token window and the same
answer plus pair-ranking objective. The shared run starts its 24 read gates
at logit `-5.3`; the one-layer control uses `-2`. All generation results use
the same held-out examples and 12-token answer limit.

| Result | One shared memory, 24 reads | One memory at layer 11 |
| --- | ---: | ---: |
| Familiar validation exact | 121/160 | 117/160 |
| Familiar validation complete pairs | 55/80 | 55/80 |
| Familiar test exact | 124/160 | 117/160 |
| Familiar test complete pairs | 58/80 | 49/80 |
| Novel-value validation exact | 2/160 | 0/160 |
| Novel-value test exact | 2/160 | 0/160 |
| Novel-value complete pairs | 0/80 | 0/80 |
| Saved and reloaded state exact | 30/40 | 31/40 |
| Saved-state complete pairs | 12/20 | 14/20 |
| Training step time | 718 s | 620 s |
| Adapter weights | 14.7 MB | 14.7 MB |

For the shared model, disabling memory and resetting it at every window both
scored 0/160 familiar validation answers. Its 200-step pilot scored 13/40
familiar answers and 3/20 complete pairs, versus 13/40 and 2/20 for a fresh
one-layer control; 24 independent adapters scored 6/40 and 0/20. Sharing is
a plausible placement choice with a small familiar-fact gain at 700 steps
and no 24-fold parameter cost. It has not learned dependable recall of
arbitrary new values.

A five-process chat check persisted two repository build tools and then
corrected one. The shared fast state answered the first question correctly
but failed the correction and second repository question: 1/3 scored answers.
This is a concrete limitation for the intended multi-fact, cross-session use.
The source-text episodic store scored 149/160 on the separate novel-value
test evaluation.

Artifacts:

- Full shared checkpoint and per-example reports: `checkpoints/session-broad-shared24-full-v1`
- Matched full-attention control: `checkpoints/session-broad-full11-scratch-v1`
- 200-step shared pilot: `checkpoints/session-broad-shared24-pilot200-v1`
- Two-step MPS smoke check: `checkpoints/session-broad-shared24-smoke-v1`
- Cross-process multi-fact check: `experiments/session-state-shared24-multifact-v1/results.json`
- Machine-readable comparison: `summary.json`

Exact train commands, dataset manifests, and report summaries are in each
checkpoint's experiment record. A different seed or natural dialogue could
change the small familiar-fact ranking.
