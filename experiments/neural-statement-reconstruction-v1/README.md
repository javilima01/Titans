# neural-statement-reconstruction-v1

68/68 exact complete statement matches; 68/68 contain the value (lenient). Source text used as query: storage diagnostic, not new-question recall.

- Neural checkpoint: `experiments/neural-statement-stress-write-v1/network.safetensors`
- Source: `experiments/neural-statement-stress-cases-v1/reconstruction.json`

## Command

```sh
.venv/bin/python -m src.llm.helpers.neural_statement_probe --cases experiments/neural-statement-stress-cases-v1/reconstruction.json --output experiments/neural-statement-reconstruction-v1 --state experiments/neural-statement-stress-write-v1/network.safetensors --read-kind passage
```

See results.json for configuration and recorded outputs. Audit source text and offline feature caches are not a persistent inference text store. This is a research probe, not default chat memory.
