# neural-statement-questions-write-v1

68 source statements written into fixed neural tensors; no recall score in this writer process.

- Neural checkpoint: `experiments/neural-statement-questions-write-v1/network.safetensors`
- Source: `experiments/neural-statement-stress-cases-v1/writes.json`

## Command

```sh
.venv/bin/python -m src.llm.helpers.neural_statement_probe --cases experiments/neural-statement-stress-cases-v1/writes.json --output experiments/neural-statement-questions-write-v1 --features 4096 --bandwidth 2 --latent-width 1024 --latent-sparsity 32 --write-only --write-questions
```

See results.json for configuration and recorded outputs. Audit source text and offline feature caches are not a persistent inference text store. This is a research probe, not default chat memory.
