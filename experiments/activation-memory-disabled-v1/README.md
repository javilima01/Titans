# activation-memory-disabled-v1

0/20 exact answer matches; 0/20 contain the value (lenient). Structured development probe, oracle supporting-statement selection.

- Neural checkpoint: `experiments/activation-memory-trained-v1`
- Source: `None`

## Command

```sh
/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python -m src.llm.helpers.activation_memory_probe evaluate --cache experiments/activation-memory-features-v1 --checkpoint experiments/activation-memory-trained-v1 --output experiments/activation-memory-disabled-v1 --eval-limit 20 --scale 0
```

See results.json for configuration and recorded outputs. Audit source text and offline feature caches are not a persistent inference text store. This is a research probe, not default chat memory.
