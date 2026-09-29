# neural-statement-attractor-v1

8/8 predictions contain the value (lenient); exact-statement scoring was not logged. Separate E5 encoder, development probe; not a Qwen activation adapter.

- Neural checkpoint: `experiments/neural-statement-attractor-v1/network.safetensors`
- Source: `experiments/neural-output-unseen-facts-v1/cases.json`

## Command

```sh
.venv/bin/python -m src.llm.helpers.neural_statement_probe --cases experiments/neural-output-unseen-facts-v1/cases.json --output experiments/neural-statement-attractor-v1 --features 4096 --bandwidth 8 --latent-width 512
```

See results.json for configuration and recorded outputs. Audit source text and offline feature caches are not a persistent inference text store. This is a research probe, not default chat memory.
