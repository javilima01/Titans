# neural-output-facts-paraphrases-v1

Network-only automatic statement writes: 5/8 independent question wordings recalled exactly.

The first paraphrase generator yielded few usable variants. Test questions did not appear among the generated write questions.

Raw inputs, predictions and available diagnostics: [results.json](results.json).

Qwen features and vocabulary decoding ran on MPS. The fixed memory update ran on CPU with FP64 covariance and FP32 network weights. No offline adapter training was used. Checkpoint files contain only network/optimizer tensors; experiment reports are never read by the inference path.
