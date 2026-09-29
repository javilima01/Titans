# neural-statement-natural-cases-v1

Manually authored before activation-memory measurements: 24 current facts, 26 important write events including two corrections, eight irrelevant utterances, 48 known and eight unknown questions. Prepared for a semantic-encoder experiment that was not run; positive statements were later evaluated with ActivationFastMemory.

Source: `experiments/neural-statement-natural-cases-v1/build_cases.py`

## Generate

```sh
.venv/bin/python -m experiments.neural-statement-natural-cases-v1.build_cases
```

Questions and expected answers in reads.json are excluded from writer inputs. Case files are benchmark/audit artifacts.
