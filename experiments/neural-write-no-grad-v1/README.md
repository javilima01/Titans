# Neural writes with autograd disabled

77 tests passed. Analytical online writes update fast weights under no_grad; frozen Titans also matches an independent autograd reference under inference_mode. These checks do not establish factual recall.

## Checks

- The linear writer matches the preconditioned gradient of an independently
  differentiated reconstruction loss after prior writes. Reconstruction error
  decreases; fast weights and optimizer state change; slow projections do not.
- With all fixed Titans parameters frozen, online outputs match the independent
  sequential autograd implementation for chunk sizes 1 and 3 under both
  `torch.no_grad()` and `torch.inference_mode()`. Fast weights keep changing
  across two windows; fixed parameters remain unchanged.
- Existing checks verify higher-order outer gradients, projection gradients,
  numerical ridge equivalence, reload/reset, and frozen-backbone training.

`no_grad()` prevents recording a reverse-mode graph. It does not prevent
explicit numerical weight updates. The linear prototype uses RLS; the original
Titans module computes MLP gradients analytically and uses momentum/forgetting.
Neither needs autograd for its online inner step. Outer training must retain
its gradient graph, as it does in the current implementation.

## Reproduce

```sh
.venv/bin/python -m unittest discover -s tests -v
```

See [results.json](results.json) and [the full output](unittest.log).
