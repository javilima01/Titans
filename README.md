# Qwen with Titans memory

`Qwen35Titans` adds trainable neural-memory adapters to a frozen Qwen3.5 model.
The adapter input/output width always matches Qwen's hidden size. The optional
`memory_hidden_size` changes only the internal memory MLP width.

```python
from src.llm.modules.qwen import Qwen35Titans

model = Qwen35Titans(
    # Omit to select CUDA, MPS, or CPU automatically.
    device="cuda",
    layer_indices=[11],  # Omit to adapt all six full-attention layers.
    memory_hidden_size=256,  # Omit to use Qwen's hidden size internally too.
    memory_chunk_size=16,
)
losses = model.train(
    train_texts=["Your training document ...", "Another document ..."],
    lr=1e-4,
    max_length=128,
    batch_size=2,
    gradient_accumulation_steps=4,
    epochs=1,
    checkpoint_decoder=True,
    loss_chunk_size=128,
)
```

The example is a starting configuration, not a measured optimum. Increase
context length and the number of adapted layers after profiling memory use on
your hardware. Smaller physical batches with more accumulation use less memory.
`losses` contains the token-mean next-token loss for each optimizer step.

## Training implementation

- Qwen weights, including its attention and output head, stay frozen. Autograd
  still propagates through Qwen activations to the adapters. Memory initialization,
  q/k/v projections, forgetting/momentum/learning-rate gates, and residual gates
  are optimized by AdamW. CUDA uses fused AdamW.
- The backbone uses BF16 by default on CUDA devices with BF16 support, otherwise
  FP32. Memory parameters, fast weights, and optimizer state stay FP32. Loading
  targets one explicit device without inference offloading.
- The memory update batches examples and tokens within chunks. Its MLP gradient
  is evaluated analytically and remains differentiable: gradients through the
  inner learning rule still reach keys, values, and gates. There is no inner
  `autograd.grad` call per token.
- A factorized read uses the outer-product structure of MLP gradients to avoid
  creating a full memory weight matrix for each token. Momentum and forgetting
  are combined using causal triangular matrices. Products avoid division by
  potentially underflowed cumulative decays.
- Memory chunks are checkpointed by default. `checkpoint_memory=False` trades
  additional saved activations for less recomputation. Whole decoder
  checkpointing is optional; enabling it disables nested memory checkpointing
  for the duration of training.
- The vocabulary projection and cross entropy are computed and checkpointed in
  `loss_chunk_size` token groups, avoiding retained logits for the entire
  sequence. The loss is the ordinary next-token objective.
- Text is tokenized once. Long documents use windows overlapping by one token,
  covering every next-token target once without mixing documents. Windows are
  independent memory episodes; context does not carry across windows.
- Shuffled length buckets reduce padding. Padding neither updates memory nor
  contributes targets. Gradients are accumulated as loss sums and normalized
  by the actual target count, including a partial final accumulation group.

## Relationship to the papers

[Titans, section 3.2](https://arxiv.org/html/2501.00663v1#S3.SS2) motivates
chunk-start gradients and matrix-based parallel updates. With chunk size 1,
this module matches the sequential surprise/momentum/forgetting recurrence.
Larger chunks evaluate all gradients at the chunk's initial memory weights,
then apply causal token updates and reads. This approximation trades some
adaptation accuracy for parallelism; it is not numerically equivalent to
refreshing the gradient after every token.

[HOPE / Nested Learning, sections 7–8](https://arxiv.org/html/2512.24695v1#S8)
motivates separating optimization levels and their update frequencies. Here,
fast memory updates occur inside each training example, while the outer language
modeling loss trains the slow adapter parameters through those updates. This
implementation remains a Titans-style adapter. HOPE's self-modifying projection
memories, continuum memory system, and specialized optimizers are not implemented.

Q/k normalization and conservative gate initialization improve the starting
numerical scale. The MLP still uses the repository's SiLU architecture and the
Qwen integration is an additive residual adapter; this is not a full paper
architecture reproduction. Quality gains require training and evaluation.

## Validation and profiling

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m tests.benchmark_titans
```

Tests compare outputs and outer gradients against an independent autograd
reference, check causality and padding state, verify the loss against Qwen's
native objective, and verify that batched/accumulated updates agree while Qwen
weights remain unchanged. A tiny hybrid Qwen model exercises the actual
Transformers integration, including decoder checkpointing and BF16 loading.

The benchmark measures memory forward + backward, not full Qwen training.
Its sequential baseline and chunked implementation use different gradient
refresh intervals; a second reference uses the same chunk update rule.
CUDA timing synchronizes the device. Full-model throughput and quality have
not been benchmarked.

One local CPU run (one thread, batch 2, 64 tokens, dimension 128, MLP width
256, chunk 16; median of three measured repetitions after warmup) produced:

| Memory forward + backward | Time |
| --- | ---: |
| Sequential autograd | 58.90 ms |
| Autograd reference with the same chunk rule | 59.33 ms |
| Factorized chunks with checkpointing | 9.03 ms |

This is a 6.5x memory-module speedup in this configuration. Hardware and tensor
sizes affect the result. A separate real Qwen3.5-0.8B smoke test completed one
training step on CPU with a BF16 backbone and one FP32 adapter (layer 23,
internal width 32, chunk 4), producing a finite loss and changing adapter
weights while the checked Qwen attention weights stayed unchanged.

Cached decoding needs a cache that carries Titans fast weights, momentum, and
chunk position alongside Qwen's cache. Until that is implemented, the Titans
wrapper defaults to `use_cache=False`, recomputing the prefix to preserve its
memory history. Explicit gradient formulas allow inference under `no_grad()`.
