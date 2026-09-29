# activation-memory-live-read-v1

15/20 exact answer matches; 19/20 contain the value (lenient). Independent processes; reader uses neural state and questions only. Structured validation cases.

- Neural checkpoint: `experiments/activation-memory-live-write-v1`
- Source: `experiments/activation-memory-live-cases-v1/reads.json`

## Command

```sh
/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python -m src.llm.helpers.activation_memory_probe session-read --checkpoint experiments/activation-memory-live-write-v1 --cases experiments/activation-memory-live-cases-v1/reads.json --output experiments/activation-memory-live-read-v1 --scale 32
```

See results.json for configuration and recorded outputs. Audit source text and offline feature caches are not a persistent inference text store. This is a research probe, not default chat memory.
