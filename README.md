# Qwen with Titans memory

`Qwen35Titans` adds trainable neural-memory adapters to a frozen Qwen3.5 model.
The adapter input/output width always matches Qwen's hidden size. The optional
`memory_hidden_size` changes only the internal memory MLP width.

## Command line workflow

`main.py` has `train`, `validate`, `test`, and `chat` subcommands. Run
`.venv/bin/python main.py COMMAND --help` for options. The default dataset is
`.datasets_cache/memory-v1` (also accepts `datasets_cache/memory-v1`), with
`train.jsonl`, `validation.jsonl`, and `test.jsonl` selected by command.

```sh
# Train adapters and save a new checkpoint directory.
.venv/bin/python main.py train --output checkpoints/memory-v1 \
  --layers 11 --memory-hidden-size 256 --window-size 512 \
  --bptt-windows 4 --gradient-accumulation-steps 4

# Generate answers on validation examples and compare memory ablations.
.venv/bin/python main.py validate --checkpoint checkpoints/memory-v1 \
  --limit 20 --ablations --report checkpoints/validation.json

# Final evaluation on the held-out test split.
.venv/bin/python main.py test --checkpoint checkpoints/memory-v1 \
  --report checkpoints/test.json

# Interactive chat with a checkpoint, or with the original Qwen model.
.venv/bin/python main.py chat --checkpoint checkpoints/memory-v1
.venv/bin/python main.py chat

# One response, then exit.
.venv/bin/python main.py chat --checkpoint checkpoints/memory-v1 \
  --prompt "Hello!" --max-new-tokens 100
```

Training writes a live report next to `--output`: `NAME.training.png` charts
(per-step training loss with validation markers, plus training vs validation
loss per epoch) and a `NAME.training.jsonl` event log. The PNG is regenerated
after every step, so opening it in a viewer that reloads changed files
(Preview, a browser) monitors training as it runs; both files move into the
checkpoint directory when training ends. Validation loss is the answer-token
cross-entropy on the training source's validation split, evaluated once per
completed epoch (`--max-steps` ends training before that epoch's validation).
`--validation-data PATH` overrides the source and `--validation-limit N`
bounds its size; sources without a validation split (a JSONL file dataset,
BABILong) monitor the training loss only. Press Ctrl+C to stop early: the
completed optimizer steps are still saved to `--output`, marked
`"interrupted": true` in the checkpoint metadata. Chart rendering requires
matplotlib in the venv:

```sh
.venv/bin/uv pip install --python .venv/bin/python matplotlib
```

Training and evaluation also update an experiment record under `experiments/`.
Each checkpoint gets a folder with `README.md` and `record.json` containing the
dataset manifest, adapter settings, training options, loss summary, and compact
evaluation metrics. [The index](experiments/README.md) links the completed and
failed runs. `validate` and `test` always save a full JSON report; when
`--report` is omitted they give it a timestamped name inside the checkpoint.
Use `--experiment-dir PATH` to keep records elsewhere. To refresh a record
after adding a report manually, run:

```sh
.venv/bin/python -m src.llm.helpers.experiment_tracking \
  --checkpoint checkpoints/session-pairs-contrastive-full-v1
```

Use `--device cpu|mps|cuda` and `--dtype float32|bfloat16` to override automatic
selection. New CLI runs default to one adapter at layer 11, internal width 256,
memory chunk size 16, Qwen window size 512, and checkpointing enabled. Start with
`train --limit 8 --max-steps 2` to check resource use. These are starting settings,
not measured performance optima. Existing checkpoints/reports are not overwritten.
Chat supports `/reset` and `/exit`; it replays conversation history per turn.
For conversations that span process launches, pass a per-user `--session-file`:

```sh
.venv/bin/python main.py chat --checkpoint checkpoints/session-profile-example \
  --session-file .sessions/alice.json
```

The session file stores the transcript after each response. The next run
replays it to reconstruct the Titans fast state. Replay cost grows with the
transcript length; `/reset` clears the saved conversation.

For compact neural state instead of replaying the full transcript, use a
per-user `--memory-state-file` with an adapter checkpoint. The file is bound to
the exact adapter weights and is updated after each answer. Reuse the same file
on later launches; `/reset` removes it. This mode keeps only the Titans fast
state, so prior messages are no longer in Qwen's ordinary attention window:

```sh
.venv/bin/python main.py chat --checkpoint checkpoints/session-broad-pairs-aligned-v1 \
  --memory-state-file .sessions/alice.safetensors --prompt "My time zone is Berlin"
.venv/bin/python main.py chat --checkpoint checkpoints/session-broad-pairs-aligned-v1 \
  --memory-state-file .sessions/alice.safetensors --prompt "What is my time zone?"
```

The compact-state mode is experimental. The synthetic training data tests
carrying facts across Qwen windows within one sequence; measure actual
cross-process recall separately before relying on it for personal or codebase
facts. A two-process smoke check recovered one trained-format build-tool fact;
in a five-process check, a correction and second repository fact were not
recalled correctly. Do not share one state file between users or adapter
checkpoints.
For a controlled held-out check, `state_evaluation` memorizes each episode's
prior sessions, saves and reloads the fast state, then asks the question with
only that state and the new session's prompt. It compares against a reset run
and updates the checkpoint's experiment record:

```sh
.venv/bin/python -m src.llm.helpers.state_evaluation \
  --data .datasets_cache/session-broad-pairs-256-v1 \
  --checkpoint checkpoints/session-broad-pairs-aligned-v1 \
  --device mps --limit 40
```

All commands accept saved normalized JSONL data via `--data PATH` (file or
directory). Training and evaluation can also download normalized episodes directly:

```sh
HF_HOME=.datasets_cache/hf .venv/bin/python main.py train \
  --hf-dataset qasper --limit 100 --output checkpoints/qasper

HF_HOME=.datasets_cache/hf .venv/bin/python main.py test \
  --hf-dataset babilong --hf-task qa1 --hf-length 4k --limit 20 \
  --checkpoint checkpoints/memory-v1
```

These support the synthetic, BABILong, and QASPER formats from the dataset helper;
arbitrary Hugging Face schemas must first be normalized. Episodes are loaded and
tokenized on CPU once; `--limit` bounds the collection. The trainer preserves whole
episodes and never discards a long document's answer. Split labels are checked to
prevent accidental training on evaluation files. BABILong has no built-in validation
split in this loader; use a separately held-out, correctly labeled JSONL subset.

### Episode training and bounded memory

The episode path uses cross-entropy on **answer tokens and optional EOS**. Plain
episodes end with `Answer:` followed by a newline; multi-session episodes use
Qwen's non-thinking assistant prefix. Both keep tokenization consistent between
training and generation. The vocabulary head runs only on
supervised positions. Context remains available for differentiable memory writes.

Qwen processes fixed windows; Titans fast weights **and momentum** carry between
windows within each episode. Qwen attention/linear-attention state and position IDs
restart each window. All state resets between episodes. `--window-size` must be a
multiple of the memory chunk size. `--bptt-windows 4` detaches state every four
windows; `0` retains the full episode graph and requires more memory. Prefix spans
without any supervised labels run without autograd because later losses cannot
cross their detach boundary. Keep relevant facts and answers within the same
backpropagation span when teaching writing/retention. Carrying detached state alone
does not teach an earlier write from a later answer loss.

With fixed windows, backpropagation span, and batch size, accelerator activation
memory is bounded with respect to episode length. CPU dataset storage and total
compute still grow. Whole-window checkpointing recomputes Qwen and its adapters
without retaining mutable state caches. Optimizer steps happen after complete
microbatches, and gradients are normalized by the actual supervised-token count.

Qwen adapters default to an inner gradient norm limit of 1.0. The norm is computed
from the factored per-token MLP gradients, includes biases, and stays differentiable.
This prevents fast-weight blowups observed during a real long-episode smoke test;
outer optimizer gradient clipping alone cannot prevent those forward-pass failures.
Python callers can configure `max_inner_grad_norm` on `Qwen35Titans` (`None` disables
it). This stability addition is saved with the checkpoint.
New adapters initialize the forgetting gate near `sigmoid(-8)` so an early write
is still present across hundreds of tokens. The inner update gate starts near
`sigmoid(-2)`, and the output branch starts near `sigmoid(-2)`. These gates remain
trainable. Existing checkpoints keep their saved gate weights.

### Checkpoint API and evaluation

```python
from src.llm.helpers.dataset_generation import read_episodes
from src.llm.modules.qwen import Qwen35Titans

model = Qwen35Titans(layer_indices=[11], memory_hidden_size=256)
losses = model.train(
    episodes=read_episodes(".datasets_cache/memory-v1/train.jsonl"),
    max_length=512,
    bptt_windows=4,
    checkpoint_decoder=True,
)
model.save_pretrained("checkpoints/my-memory")
loaded = Qwen35Titans.from_pretrained("checkpoints/my-memory")
```

Checkpoints contain adapter/gate weights in Safetensors, architecture settings,
training window settings, and the tokenizer; CLI training runs also add the
monitor report (`training.png`, `training_log.jsonl`) and record the loss
history in the metadata. Loading reconstructs the adapters and checks tensor
names/shapes against the saved architecture. The unchanged `Qwen/Qwen3.5-0.8B`
base weights are reused from the model cache (downloaded if absent). Optimizer
state and individual episodes' fast memory are not saved.
`train --checkpoint OLD --output NEW` continues training with a fresh optimizer.

Validation/test greedily generate answers with the saved window size and report
normalized exact match, word-token precision/recall/F1, and per-task scores.
Recall means answer-token recall, not retrieval Recall@K. Alternative answer
annotations are scored against the best reference by F1. JSON reports include
individual predictions; metric summaries go to stdout and progress to stderr.
`--ablations` additionally disables the memory branch and resets memory at every
window. These use the same bounded Qwen windows, so the disabled condition is a
windowed baseline. Training and evaluation use the same streaming state path;
each generated partial window is replayed from its starting state to avoid
writing the same tokens into memory twice.

The legacy plain-text training API remains available:

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

## Memory datasets

`src/llm/helpers/dataset_generation.py` generates synthetic episodes and downloads
selected [BABILong](https://github.com/booydar/babilong) or
[QASPER](https://huggingface.co/datasets/allenai/qasper) data from Hugging Face.
Each JSONL record contains one independent episode: `id`, ordered `context`,
`question`, alternative accepted `answers`, `source`, `split`, and `metadata`.
Data stays as text, so generation requires neither Qwen nor a tokenizer.

### Generate synthetic data

```sh
.venv/bin/python -m src.llm.helpers.dataset_generation synthetic \
  --output .datasets_cache/memory-v1 \
  --train-size 1000 --validation-size 100 --test-size 100 \
  --seed 42 --num-entities 8 \
  --distractor-sentences 128 --min-tail-sentences 32
```

This creates `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `manifest.json`.
All three task types are included by default:

- `recall`: retrieve a fresh random station/code association.
- `update`: retrieve the latest code after corrections to multiple stations.
- `multi_hop`: follow a courier-to-station association and retrieve that station's code.

Use `--tasks recall` for the initial easy curriculum, then add `update multi_hop`.
Facts are interspersed with generated distractors; `min_tail_sentences` controls
the minimum number of distractor sentences after the last fact. These lengths
are **sentence counts, not token counts**. Measure them with the Qwen tokenizer
when choosing training windows. Synthetic metadata includes supporting-fact
character spans for measuring actual token distances later.

Generation is deterministic per seed/config/split/episode index. Splits use
independent random associations but share task templates; they test new facts,
not generalization to unseen task templates. Synthetic filler is deliberately
simple. Add real-document tasks after recall works. Output directories must be
new; existing datasets are never silently overwritten.

### Generate multi-session training episodes

`synthetic-sessions` creates Qwen chat-formatted histories with a fact in session 1,
an optional correction in session 2, unrelated turns, and a question in session 3.
It supports names, ages, preferences, time zones, editors, current projects,
repository entry points, runtimes, test commands, config files, build tools,
and corrections to ages, entry points, and test commands.
The generated splits have independent users and repositories. They supervise
answer tokens without EOS, so a low loss cannot come mainly from learning when
to stop. For example:

```sh
.venv/bin/python -m src.llm.helpers.dataset_generation synthetic-sessions \
  --output .datasets_cache/session-profile-answer-only-v2 \
  --train-size 1200 --validation-size 160 --test-size 160 \
  --seed 44 --gap-turns 7 --tasks name age preference age_update

.venv/bin/python main.py train --data .datasets_cache/session-profile-answer-only-v2 \
  --output checkpoints/session-profile-example --layers 11 --window-size 256 \
  --validation-limit 40 --gradient-accumulation-steps 4 \
  --lr 0.0003 --no-checkpoint-decoder
```

For harder training, `synthetic-session-pairs` changes one earlier fact while
holding the question and trailing conversation fixed. Each adjacent pair has
two different correct answers. Keep the pair in one optimizer step with
`--batch-size 2 --no-shuffle`:

```sh
.venv/bin/python -m src.llm.helpers.dataset_generation synthetic-session-pairs \
  --output .datasets_cache/session-pairs-256-v1 \
  --train-pairs 600 --validation-pairs 60 --test-pairs 60 \
  --seed 47 --gap-turns 7 \
  --tasks name age preference repo_entry age_update repo_update

.venv/bin/python main.py train --data .datasets_cache/session-pairs-256-v1 \
  --output checkpoints/session-pairs-256-v1 --layers 11 --window-size 256 \
  --memory-qk-scale 5.656854249492381 --aligned-qk-init \
  --bptt-windows 4 --batch-size 2 --no-shuffle --lr 0.001

.venv/bin/python main.py train --data .datasets_cache/session-pairs-256-v1 \
  --checkpoint checkpoints/session-pairs-256-v1 \
  --output checkpoints/session-pairs-contrastive-v1 --window-size 256 \
  --bptt-windows 4 --batch-size 2 --no-shuffle \
  --pair-contrastive-weight 16 --lr 0.0003
```

There are 1,200 training episodes (600 pairs). With the saved Qwen tokenizer,
both prompts in every pair have the same final 256 tokens, the changed fact is
at least 277 tokens before the answer, and each prompt fits in 512 tokens.
Measure held-out paired accuracy and compare `normal` with `reset` and
`disabled`; a lower training loss alone does not establish recall.
For a broader fact mix, omit `--tasks`; the current defaults include user
preferences and several repository facts:

```sh
.venv/bin/python -m src.llm.helpers.dataset_generation synthetic-session-pairs \
  --output .datasets_cache/session-broad-pairs-256-v1 \
  --train-pairs 700 --validation-pairs 80 --test-pairs 80 \
  --seed 61 --gap-turns 7
```

This produces 1,400 training episodes. The changed supporting fact is 276–312
tokens before the question on the saved tokenizer, and each pair has identical
final 256 prompt tokens. The templates are a synthetic recall probe; success
on them does not establish recall of arbitrary real user or codebase facts.
Use `novel_value_dataset` to replace only held-out answers with values absent
from the training pools while keeping the questions and surrounding sessions
fixed:

```sh
.venv/bin/python -m src.llm.helpers.novel_value_dataset \
  --source .datasets_cache/session-broad-pairs-256-v1 \
  --output .datasets_cache/session-broad-novel-values-v1 --seed 71
```

The separate `session-unseen-pairs-256-v1` dataset holds out three entire fact
categories: work hours, deployment region, and release branch. Together these
two checks distinguish recall of familiar answer choices from transfer to new
values and fact types.

```sh
.venv/bin/python -m src.llm.helpers.dataset_generation synthetic-session-pairs \
  --output .datasets_cache/session-unseen-pairs-256-v1 \
  --train-pairs 0 --validation-pairs 80 --test-pairs 80 \
  --seed 67 --gap-turns 7 \
  --tasks user_work_hours repo_deploy_region repo_release_branch
```

To compare neural memory with a simple text-retrieval control, run
`retrieval_evaluation` on the same held-out episodes. It ranks earlier user
messages by lexical overlap with the question and places the selected message
next to the question with Titans memory disabled. The `oracle` mode uses the
dataset's marked support span as an upper bound. Both scores and the retrieval
hit rate are saved in the checkpoint report and `experiments/` record:

```sh
.venv/bin/python -m src.llm.helpers.retrieval_evaluation \
  --data .datasets_cache/session-broad-novel-values-v1 \
  --checkpoint checkpoints/session-broad-pairs-aligned-v1 \
  --device mps --limit 40
```

The lexical baseline is easiest on these templated conversations; evaluate
realistic paraphrases, conflicts, and unrelated facts before treating its
synthetic hit rate as a product result.

The optional second stage adds a ranking loss at the first answer token that
differs within each pair. It compares both candidate tokens under both
histories, so an answer prior shared by the pair cancels. `--batch-size 2` and
`--no-shuffle` are required to preserve pair alignment. Training loss includes
this extra term; validation loss remains answer-token cross-entropy, and
`paired_metrics.both_correct_rate` in generation reports is the direct recall
check. `--supervise-eos` also trains the end-of-answer token when short answers
do not stop cleanly. Later answer tokens are excluded from the ranking loss
because teacher forcing can expose an earlier differing answer token.
Query/key scaling and aligned initialization apply only to new adapters;
saved older checkpoints retain their original settings.
On Apple Silicon, add `--device mps --no-checkpoint-decoder` when PyTorch reports
MPS available. A restricted execution sandbox may hide the GPU even on a
supported Mac; check `.venv/bin/python -c 'import torch; print(torch.backends.mps.is_available())'`
in the same environment that will run training.

`gap_turns` counts filler user/assistant exchanges, not tokens. With the saved Qwen
tokenizer, every latest fact in this dataset is at least 277 tokens before the
answer, and every prompt is under 512 tokens. This trains and tests transfer
across a 256-token window boundary. For all six tasks at a 512-token window,
use `--gap-turns 15`; the generated examples have the latest fact at least 541
tokens before the answer. Each record remains one
memory lifetime; training carries Titans state between windows inside that
record. A checkpoint does not store per-user facts or fast state. Use
`chat --session-file` to keep a per-user transcript and reconstruct fast state
across process launches. A production service would need equivalent per-user
storage and should process new turns incrementally to avoid replay cost.

For a later stage that also trains repository entry points and corrections, include
`repo_entry repo_update` in `--tasks`. The profile-only first stage keeps answer
lengths similar while testing whether cross-window recall works at all.

### Download selected Hugging Face data

The optional downloader was tested with `datasets==5.0.1`:

```sh
.venv/bin/uv pip install --python .venv/bin/python 'datasets==5.0.1'

HF_HOME=.datasets_cache/hf .venv/bin/python -m src.llm.helpers.dataset_generation huggingface \
  --dataset babilong --split train --length 1k --task qa1 --limit 1000 \
  --cache-dir .datasets_cache/hf --output .datasets_cache/babilong/train-qa1-1k.jsonl

HF_HOME=.datasets_cache/hf .venv/bin/python -m src.llm.helpers.dataset_generation huggingface \
  --dataset qasper --split train --limit 1000 \
  --cache-dir .datasets_cache/hf --output .datasets_cache/qasper/train.jsonl
```

Downloads stream only the selected task/split files and stop after `--limit`
episodes (or source exhaustion). This limits emitted examples, not network
bytes: reading a JSON file or Parquet row group may fetch additional data.
The resulting JSONL is self-contained and can be read offline. Each episode
records the resolved source commit and file paths; pass `--revision COMMIT`
to repeat a download at the same revision.

BABILong `train` uses `RMT-team/babilong-train-5k-samples`; `test` uses the
separate `RMT-team/babilong` evaluation repository. There is no official
validation split in this loader. Keep a separate held-out training subset when
using BABILong alone. Not every task/length exists upstream; missing combinations
raise an error. We select files directly because the upstream dataset card
includes missing task files. QASPER preserves official document-level splits,
uses the `refs/convert/parquet` export by default, and includes paper text and
figure/table captions. It retains alternative answer annotations, including
yes/no and unanswerable cases; questions without answer annotations are skipped.

### Read episodes and inspect answer labels

```python
from src.llm.helpers.dataset_generation import read_episodes

episode = next(read_episodes(".datasets_cache/memory-v1/train.jsonl"))
prompt = episode.prompt  # Context + question; excludes the target answer.
example = episode.training_example()
text = example["text"]
start, end = example["answer_span"]  # Half-open CHARACTER offsets, not token IDs.
assert text[start:end] == episode.answers[0]
```

The renderer uses the first accepted answer as its training target. Tokenize
the complete rendered text with offset mappings to derive token loss masks;
tokenizing prompt and answer separately can change their BPE boundary. Metadata
must never become model input. Preserve context order and reset memory between
episodes; QASPER `document_id` is available for grouping related questions.

Use `train(episodes=...)` for this schema. The legacy `train(train_texts=...)`
path uses independent windows and loss on all next tokens.

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
  for the duration of training. Episode training checkpoints whole Qwen windows,
  including their explicit returned memory state.
- The vocabulary projection and cross entropy are computed and checkpointed in
  `loss_chunk_size` token groups, avoiding retained logits for the entire
  sequence. The loss is the ordinary next-token objective.
- Legacy plain text is tokenized once. Long documents use windows overlapping by one token,
  covering every next-token target once without mixing documents. Windows are
  independent memory episodes; context does not carry across windows.
- Shuffled length buckets reduce padding. Padding neither updates memory nor
  contributes targets. Gradients are accumulated as loss sums and normalized
  by the actual target count, including a partial final accumulation group.

## Relationship to the papers

[Titans, section 3.2](https://arxiv.org/html/2501.00663v1#S3.SS2) motivates
chunk-start gradients and matrix-based parallel updates. With chunk size 1 and
inner gradient clipping disabled, this module matches the sequential
surprise/momentum/forgetting recurrence.
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

The episode/CLI tests additionally cover carried-state gradients, differentiable
inner clipping, answer/EOS masks, checkpoint round trips, generation across window
boundaries, metrics, and all four commands. A real Qwen smoke test on one full
`memory-v1` episode completed with BF16 base weights, the CLI's default layer 11 /
width 256 / window 512 settings, and a finite loss of 3.5707. Checkpoint loading,
validation, and chat were also exercised with a smaller one-step checkpoint.
These are functional checks, not evidence of trained recall quality; the full
1,000-episode training run has not been performed.

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

Generation now carries Titans state explicitly between complete Qwen windows
and recomputes only the current partial window for each generated token.
HF KV caching remains disabled for the Titans path; each window restarts Qwen's
own attention state. Explicit gradient formulas allow inference under `no_grad()`.
