# activation-memory-natural-read-v1

3/48 exact answer matches; 3/48 contain the value (lenient). Unknown questions: 1/8 empty outputs. Independent processes, hand-selected statements. General fact recall failed.

- Neural checkpoint: `experiments/activation-memory-natural-write-v1`
- Source: `experiments/activation-memory-natural-cases-v1/reads.json`

## Command

```sh
/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python -m src.llm.helpers.activation_memory_probe session-read --checkpoint experiments/activation-memory-natural-write-v1 --cases experiments/activation-memory-natural-cases-v1/reads.json --output experiments/activation-memory-natural-read-v1 --scale 32
```

See results.json for configuration and recorded outputs. Audit source text and offline feature caches are not a persistent inference text store. This is a research probe, not default chat memory.
