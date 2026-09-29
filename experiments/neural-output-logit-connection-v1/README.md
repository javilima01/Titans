# neural-output-logit-connection-v1

Network-only output connection: base-only 0/8; scales 8, 16, 32 gave 1/8, 5/8, 7/8.

A parallel addition of memory cosine scores to frozen LM logits improved the development probe. The remaining result was Meson followed by extra text. Scale 32 was selected on this development probe; it is not a held-out score.

Raw inputs, predictions and available diagnostics: [results.json](results.json).

Qwen features and vocabulary decoding ran on MPS. The fixed memory update ran on CPU with FP64 covariance and FP32 network weights. No offline adapter training was used. Checkpoint files contain only network/optimizer tensors; experiment reports are never read by the inference path.

## Reproduce

Use a new output path; existing reports are protected.

```sh
.venv/bin/python -m src.llm.helpers.neural_connection_probe --state experiments/neural-output-facts-paraphrases-v2/network.safetensors --output experiments/NEW-RUN/results.json
```
