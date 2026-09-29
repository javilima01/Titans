# neural-output-memory-v1

Network-only memory: 40/40 exact demonstrated answers, including 20 corrected values, with one write per demonstration.

Twenty questions coexist in one fixed network. After each write phase only weight and inverse_covariance tensors were checkpointed and reloaded. The reader receives the question alone. No question index, strings, stored examples or target-token table are retained by the memory.

Raw inputs, predictions and available diagnostics: [results.json](results.json).

Qwen features and vocabulary decoding ran on MPS. The fixed memory update ran on CPU with FP64 covariance and FP32 network weights. No offline adapter training was used. Checkpoint files contain only network/optimizer tensors; experiment reports are never read by the inference path.

## Reproduce

Use a new output path; existing reports are protected.

```sh
.venv/bin/python -m src.llm.helpers.neural_memory_probe --output experiments/NEW-RUN/results.json
```
