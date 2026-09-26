# AGENTS.md

Qwen3.5-0.8B with trainable Titans-style neural-memory adapters. `README.md` is the design reference (training algorithm, paper relationship, benchmark numbers); read it before changing `src/llm/modules/titans.py`.

## Commands (from repo root)

- All tests: `.venv/bin/python -m unittest discover -s tests -v`
- Single test: `.venv/bin/python -m unittest tests.test_titans_training.MemoryTests.test_checkpoint_matches_eager`
- Benchmark: `.venv/bin/python -m tests.benchmark_titans [--device mps]`
- Lint/format: `.venv/bin/ruff check .` and `.venv/bin/ruff format .`

## Environment gotchas

- Imports are absolute `src.llm...` (`src/` is a namespace package), so Python commands must run from the repo root.
- `pyproject.toml` holds only Ruff config - there is no `[project]` table. The pre-provisioned `.venv` (Python 3.14, torch 2.14, transformers 5.17) is the source of truth; `uv sync` will not rebuild dependencies.
- `uv` is vendored at `.venv/bin/uv`, not on PATH. The pre-commit hook (`.githooks/pre-commit`, enabled via `git config core.hooksPath=.githooks`) runs `uv run ... ruff format --check` and `ruff check` on staged Python files, so commits fail unless `.venv/bin` is on PATH.
- `Qwen35Wrapper` downloads `Qwen/Qwen3.5-0.8B` to `.models_cache/Qwen/Qwen3.5-0.8B` on first use (~1.7 GB, network), then loads with `local_files_only=True`. The unittest suite uses tiny in-memory models and needs no download.
- `qwen.py` sets `HF_DEACTIVATE_ASYNC_LOAD=1` at import before importing transformers; keep that ordering.

## Architecture

- `titans.py`: `TitansMemory` chunked causal recurrence (factorized gradients, optional checkpointing); `AttentionWithTitans` adds the gated additive memory branch to a Qwen attention module.
- `qwen.py`: `Qwen35Wrapper` is frozen inference; `Qwen35Titans` installs adapters only on full-attention layers (those with `self_attn`; Qwen3.5 is a hybrid linear/full-attention model, so `layer_indices` must be full-attention) and `train()` optimizes only memory and gates.
- `training.py`: tokenize-once overlapping windows (one token overlap), length-bucketed batching, chunked cross-entropy that never materializes full logits.
- Memory adapters stay FP32 even with a BF16 backbone; `TitansMemory.forward` deliberately disables autocast. Titans state is not in HF's KV cache, so the wrapper forces `use_cache=False` during generation.

## Testing

- `tests/test_titans_training.py:reference_memory` is an independent sequential-autograd implementation of the chunk-start gradient rule. Value/gradient tests and the benchmark compare against it; any `TitansMemory` change must preserve this numerical equivalence and the reported speedup.
- Tests are CPU-only, single-threaded (`torch.set_num_threads(1)`), seeded, and finish in well under a second.
- Ruff is strict (py314, 100 cols, double quotes, no relative imports); tests are exempt from `ARG` and `SLF001`, so accessing private members there is expected.
