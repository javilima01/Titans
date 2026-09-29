# neural-statement-stress-cases-v1

Synthetic development set: 64 current facts, 68 writes including four corrections, 128 known questions and 16 unknown questions. Seed 9041. reconstruction.json uses the 68 source statements as cues for a separate storage diagnostic.

Source: `src/llm/helpers/neural_statement_cases.py`

## Generate

```sh
.venv/bin/python -m src.llm.helpers.neural_statement_cases --output experiments/neural-statement-stress-cases-v1 --seed 9041
```

Questions and expected answers in reads.json are excluded from writer inputs. Case files are benchmark/audit artifacts.
