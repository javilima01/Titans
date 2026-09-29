# neural-statement-addressing-v1

Offline addressing diagnostic with access to source vectors. Cosine and exact-kernel comparisons are diagnostic upper bounds, not admissible network-only readers. Random-feature coefficient ranking is also only a diagnostic, not decoded sentence accuracy.

- Neural checkpoint: `None`
- Source: `experiments/neural-statement-stress-cases-v1`

## Command

```sh
.venv/bin/python -m experiments.neural-statement-addressing-v1.diagnostic
```

This analysis retains source vectors solely to diagnose addressing. Its ranking scores are not generated-answer accuracy and it is not an admissible inference implementation.
