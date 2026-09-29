# activation-memory-trained-multi4-v1

Slow read/write projections trained through an inner fast-weight solve on observed statement tokens. No test-time question/answer demonstrations or separate semantic encoder. Generation still requires independent evaluation; address accuracy is not answer accuracy.

- Neural checkpoint: `experiments/activation-memory-trained-multi4-v1/network.safetensors`
- Source: `experiments/activation-memory-features-v1`

## Command

```sh
/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python -m src.llm.helpers.activation_memory_probe train --cache experiments/activation-memory-features-v1 --output experiments/activation-memory-trained-multi4-v1 --epochs 12 --facts-per-memory 4
```

See results.json for configuration and recorded outputs. Audit source text and offline feature caches are not a persistent inference text store. This is a research probe, not default chat memory.
