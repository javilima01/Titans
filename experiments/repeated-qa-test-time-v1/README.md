# Five repeated Q/A demonstrations at test time

The 700-step shared surprise-delta checkpoint was given the exact question and
answer from each evaluation example either zero, one, or five times. Each
example started with an empty fast state. For one and five presentations, the
Q/A chat text was processed by `memorize_text`, then the resulting state was
saved to Safetensors and reloaded. The final generation was a new session with
only the question and a generic answer-format system instruction. No answer
text or prior dialogue was in that prompt. Twenty counterfactual pairs (40
examples) used answer values absent from adapter training.

| Test on 40 unseen values | Exact answers | Complete pairs |
| --- | ---: | ---: |
| Question only, empty state | 0 | 0/20 |
| One Q/A, saved state, question only | 0 | 0/20 |
| Five Q/A, saved state, question only | 0 | 0/20 |
| Five Q/A visible in 512-token context, memory disabled | 40 | 20/20 |

All five demonstrations fit in the direct-context window (205–273 tokens).
The same saved-state test also scored 0/40 with five presentations of answer
values present in adapter training. Removing the generic system instruction
from the final prompt left the unseen-value result at 0/40. Repetition changed
the predictions, but did not yield the exact demonstrated values. The
direct-context control shows Qwen can copy these Q/A answers when it sees
their text. This result does **not** establish that more than five exposures,
new Q/A-specific training, or a different memory connection would fail.

Reproduce with `src/llm/helpers/repetition_probe.py`. The source JSON reports
are in `checkpoints/surprise-delta-open-values-700-v1/`:

- `repeated-qa-novel-system-validation-40.json` — main 0/1/5 comparison
- `repeated-qa-seen-system-validation-40.json` — seen-value comparison
- `repeated-qa-direct-context-novel-40.json` — direct-context control
- `repeated-qa-novel-validation-40.json` — question-only without system instruction

The machine-readable summary is `results.json` in this folder.
