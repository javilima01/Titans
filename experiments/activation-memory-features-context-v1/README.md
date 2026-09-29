# activation-memory-features-context-v1

Adds causal token-context activations to test whether identity information was lost before the fast-memory writer/read projections. No explicit entity extraction or stored identity index.

- Neural checkpoint: `None`
- Source: `None`

## Command

```sh
/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python -m src.llm.helpers.activation_memory_probe augment-cache --cache experiments/activation-memory-features-v1 --output experiments/activation-memory-features-context-v1
```

See results.json for configuration and recorded outputs. Audit source text and offline feature caches are not a persistent inference text store. This is a research probe, not default chat memory.
