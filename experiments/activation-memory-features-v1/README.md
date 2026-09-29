# activation-memory-features-v1

Feature cache for outer learning. Source statements and teacher-forced query prefixes are training/evaluation artifacts, never persistent inference memory. The latest supporting statement is selected by benchmark annotations.

- Neural checkpoint: `None`
- Source: `.datasets_cache/session-open-values-v1`

## Command

```sh
/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python -m src.llm.helpers.activation_memory_probe cache --output experiments/activation-memory-features-v1 --train-limit 1200 --eval-limit 80
```

See results.json for configuration and recorded outputs. Audit source text and offline feature caches are not a persistent inference text store. This is a research probe, not default chat memory.
