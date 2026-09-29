# neural-output-facts-paraphrases-v2

Network-only automatic writes with demonstrated paraphrasing: 6/8 exact reads.

All eight read question strings were absent from write supervision. Some generated paraphrases change the requested property; source substring validation cannot detect this.

Raw inputs, predictions and available diagnostics: [results.json](results.json).

Qwen features and vocabulary decoding ran on MPS. The fixed memory update ran on CPU with FP64 covariance and FP32 network weights. No offline adapter training was used. Checkpoint files contain only network/optimizer tensors; experiment reports are never read by the inference path.

## Reproduce

Use a new output path; existing reports are protected.

```sh
.venv/bin/python -m src.llm.helpers.neural_fact_probe --paraphrases --output experiments/NEW-RUN/results.json
```
