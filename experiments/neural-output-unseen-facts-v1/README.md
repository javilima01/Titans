# neural-output-unseen-facts-v1

Independent new facts: 4/8 exact answers with direct neural readout.

The writer reversed the configuration relation, omitted the dietary fact, and generated answer-bearing keys for the launch command. The reader also confused the maintainer with the release branch. No read question occurred in write supervision. This is an unresolved generalization failure; the network-only design is experimental.

Raw inputs, predictions and available diagnostics: [results.json](results.json).

Qwen features and vocabulary decoding ran on MPS. The fixed memory update ran on CPU with FP64 covariance and FP32 network weights. No offline adapter training was used. Checkpoint files contain only network/optimizer tensors; experiment reports are never read by the inference path.

## Reproduce

Use a new output path; existing reports are protected.

```sh
.venv/bin/python -m src.llm.helpers.neural_fact_probe --paraphrases --cases experiments/neural-output-unseen-facts-v1/cases.json --output experiments/NEW-RUN/results.json
```
