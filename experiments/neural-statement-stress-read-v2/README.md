# neural-statement-stress-read-v2

61/128 exact complete statement matches; 69/128 contain the value (lenient). Unknown questions: 0/16 empty outputs. Separate E5 encoder, development probe; not a Qwen activation adapter.

- Neural checkpoint: `experiments/neural-statement-stress-write-v2/network.safetensors`
- Source: `experiments/neural-statement-stress-cases-v1/reads.json`

## Command

```sh
.venv/bin/python -m src.llm.helpers.neural_statement_probe --cases experiments/neural-statement-stress-cases-v1/reads.json --output experiments/neural-statement-stress-read-v2 --state experiments/neural-statement-stress-write-v2/network.safetensors
```

See results.json for configuration and recorded outputs. Audit source text and offline feature caches are not a persistent inference text store. This is a research probe, not default chat memory.
