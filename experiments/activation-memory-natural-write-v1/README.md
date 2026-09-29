# activation-memory-natural-write-v1

26 source statements written into fixed neural tensors; no recall score in this writer process.

- Neural checkpoint: `experiments/activation-memory-natural-write-v1/network.safetensors`
- Source: `experiments/activation-memory-natural-cases-v1/writes.json`

## Command

```sh
/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python -m src.llm.helpers.activation_memory_probe session-write --checkpoint experiments/activation-memory-trained-multi20-v2 --cases experiments/activation-memory-natural-cases-v1/writes.json --output experiments/activation-memory-natural-write-v1
```

See results.json for configuration and recorded outputs. Audit source text and offline feature caches are not a persistent inference text store. This is a research probe, not default chat memory.
