# Memory research notes

## Current constraint: facts must reside in the network

The user clarified that the memory must be the network itself. The tentative
question-index/replay/text-readout implementation was removed from the active
code and CLI; its failed and successful probes remain archived for audit.
Earlier source-text experiments below are historical controls, not the
implementation being pursued under this constraint.

## Current direction: Qwen activations update neural weights

`ActivationFastMemory` consumes the frozen decoder's final activation and a
normalized mean of its 24 block outputs (attention and DeltaNet). Separate
learned projections produce write keys and read queries. Observed next-token
embeddings supervise a linear fast network. Only that network's weights,
its inverse-covariance optimizer tensor, and the learned slow projections
are needed in a later process. Facts are not persisted as examples or text.

This is an experimental linear test-time learner. It is related to the idea
of treating recurrent state as an adapting model in
[Test-Time Training](https://arxiv.org/abs/2407.04620), and to the neural-memory
motivation in [Titans](https://arxiv.org/abs/2501.00663). It does not reproduce
the full Titans or HOPE architecture: its online optimizer is recursive least
squares, without Titans' learned momentum and forgetting. The original
Titans module remains available and unchanged.

### Why writes still learn under `torch.no_grad()`

PyTorch's [`no_grad`](https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html)
disables recording operations for reverse-mode differentiation. The code can
still compute an analytical loss gradient and explicitly update weights.
For one row key `k`, target embedding `v`, fast matrix `W`, and optimizer `P`:

```text
loss = 0.5 * ||k W - v||²
gradient = k.T (k W - v)
W_next = W - P gradient / (1 + k P k.T)
```

`write()` implements the equivalent block RLS update `W += gain @ error`
and updates `P`. Its `no_grad()` decorator prevents building an unnecessary
graph through the history of online writes. It does not freeze `W`.
Outer training uses the differentiable `episode_read()` solve, with autograd
enabled, to learn the slow read/write projections. Qwen stays frozen.

The original `TitansMemory` instead computes the memory MLP's gradient by
the explicit chain rule, then applies learned momentum and forgetting to
the carried fast-state tensors. This also works under `no_grad()` or
`inference_mode()`. The frozen initial MLP parameters are distinct from its
evolving fast weights. If a future writer uses `torch.autograd.grad()` to
compute its inner gradient, that loss computation must run with gradients
enabled; the current analytical implementations do not need that graph.

[The gradient verification](neural-write-no-grad-v1/README.md) checks the
linear write against a preconditioned autograd gradient, and the frozen
Titans recurrence against an independent sequential autograd reference.
It also checks state continuation across windows, unchanged slow parameters,
and outer gradients reaching both learned projections. These are numerical
correctness checks, not a demonstration of reliable factual recall.

### Training and evaluation protocol

- Cache 1,200 training episodes and 80 each from validation and test in
  [activation-memory-features-v1](activation-memory-features-v1/README.md).
  Answer strings, account IDs, and repository IDs in these evaluation subsets
  do not overlap the training subset. Templates and fact categories do overlap.
- Select each episode's latest supporting statement using dataset annotations.
  This is oracle importance selection; the writer has not learned which facts
  to keep. Activations before each observed token predict its frozen output
  embedding. No evaluation question or answer demonstration is written.
- Train read/write projections with a differentiable inner ridge solve,
  embedding reconstruction, and an auxiliary token-addressing loss. Simulated
  memories contain 1, 4, or 20 facts. Counterfactual variants of the same
  question are kept in separate banks. Checkpoints are selected by validation
  loss; model configurations are compared using validation exact generation.
- Evaluate with the new question and generated prefix only, after saving and
  reloading the neural tensors. Add memory cosine scores to frozen Qwen logits
  at scale 32 unless indicated. Generation is greedy, up to 24 tokens.
- Run Qwen on MPS, small projection training and FP64 online solves on CPU.
  A 1,024-wide model occupies about 36 MiB: 24 MiB slow projections and
  12 MiB fast weights plus optimizer. Offline feature caches are training
  artifacts, not inputs to the separate live reader.

### Results

"Value present" only checks a substring; repetitions, extra claims, and wrong
surrounding text can still make those answers unusable. Exact match is the
primary comparison. Each entry below has 80 questions unless stated otherwise.

| Run | Training facts per bank / key width / epochs | Evaluation facts per bank | Exact | Value present |
| --- | --- | ---: | ---: | ---: |
| [Validation, memory scores only](activation-memory-validation-v1/README.md) | 1 / 256 / 12 | 1 | 58/80 | 80/80 |
| [Validation, add base logits](activation-memory-validation-logits32-v1/README.md) | 1 / 256 / 12 | 1 | 69/80 | 80/80 |
| [Validation, more coexisting facts](activation-memory-validation-multi20-v1/README.md) | 1 / 256 / 12 | 20 | 20/80 | 20/80 |
| [Validation, train with 4 facts](activation-memory-validation-multi20-v2/README.md) | 4 / 256 / 12 | 20 | 41/80 | 48/80 |
| [Validation, larger memory and 20 facts](activation-memory-validation-multi20-v3/README.md) | 20 / 1024 / 12 | 20 | 57/80 | 65/80 |
| [Validation, more training](activation-memory-validation-multi20-v4/README.md) | 20 / 1024 / 60 | 20 | 64/80 | 74/80 |
| [Validation, add causal token pooling](activation-memory-validation-context-v1/README.md) | 20 / 1024 / 60 | 20 | 62/80 | 75/80 |
| [Held-out test, single fact](activation-memory-test-single-v1/README.md) | 1 / 256 / 12 | 1 | 66/80 | 80/80 |
| [Held-out test, coexisting facts](activation-memory-test-multi20-v1/README.md) | 20 / 1024 / 60 | 20 | 56/80 | 68/80 |
| [Validation, memory disabled](activation-memory-disabled-v1/README.md) | 20 / 1024 / 60 | 20 | 0/20 | 0/20 |

The 60-epoch model selected epoch 48 by validation loss. The token-pooling
variant selected epoch 55 and had lower reconstruction loss, but did not
improve validation exact generation. It was not selected for the primary
multi-fact test. Increasing bank size and key width together confounds their
individual effects; this sequence is not a matched scaling ablation.

A [separate-process live write](activation-memory-live-write-v1/README.md)
ingested 20 raw statements, with no cached features or answer annotations.
The [fresh reader](activation-memory-live-read-v1/README.md) loaded only
neural state and questions: **15/20 exact, 19/20 with the value present**.
These statements came from the structured validation subset. One wrong value
was `config/queue.ini` instead of `settings/web.yaml`.

### Broader language remains a failure

The [natural-language probe](activation-memory-natural-read-v1/README.md)
contains 24 current user/codebase facts, 26 positive statements including two
corrections, 48 known questions and eight unknown questions. Its source cases
were authored before measuring this activation-memory architecture. Negative
utterances were excluded using human labels: selection was still an oracle.
The writer and reader ran as independent processes.

Only **3/48 known questions** were correct (also 3/48 for value presence):
two questions about a project codename and one temperature preference.
**1/8 unknown questions** produced empty output; no reliable abstention
mechanism was established. File paths, commands, paraphrases, and corrections
largely failed. This model therefore does not satisfy general user/codebase
recall, despite its better structured-template results.

The next hypothesis to test is broader outer training that explicitly teaches
statement-to-question addressing for varied wording, code observations,
corrections, and unrelated coexisting facts. The current evidence does not
establish whether that will be sufficient. Keep these failed cases as
development cases and create a fresh independent evaluation before tuning.

## Separate semantic-encoder diagnostics

Before returning to Qwen activations, `NeuralStatementMemory` tested a frozen
[E5 encoder](https://huggingface.co/intfloat/e5-small-v2), random Fourier
features, fixed dense online weights, and a byte decoder. An optional recurrent
associative network cleans a latent code before decoding a whole statement.
The neural state has fixed size and the reader does not retrieve stored text,
but this separate encoder is not the requested attention-driven Qwen adapter.

The initial eight-case development probe improved from 3/8 to 7/8 value
presence as features/encoding changed, then reconstructed 8/8 whole statements
with a recurrent cleanup network. The later stress set contains 64 current
facts across four repositories and user properties, four corrections,
128 known questions and 16 unknown questions. The same probe was reused
for these architecture comparisons; they are development results.

| Reader run | Complete correct statement, known questions | Empty output, unknown questions |
| --- | ---: | ---: |
| [Dense latent, bandwidth 8](neural-statement-stress-read-v1/README.md) | 48/128 | 0/16 |
| [Dense latent, bandwidth 2](neural-statement-stress-read-v2/README.md) | 61/128 | 0/16 |
| [Sparse latent](neural-statement-sparse-read-v1/README.md) | 89/128 | 0/16 |
| [Temporary write-side generated questions](neural-statement-questions-read-v1/README.md) | 113/128 | 0/16 |
| [Normalize correction prompts and retain repository names](neural-statement-questions-read-v2/README.md) | 117/128 | 0/16 |

Using the original source itself as the cue reconstructed **68/68** written
statements. This is a storage diagnostic, not new-question recall. An offline
addressing analysis retained source vectors solely to diagnose representation
errors; it is not an admissible neural-memory inference implementation.
The larger sparse model occupies about 206 MiB. Proposed importance selection
and read verification flags were implemented but not evaluated. The separate
encoder was not run on the natural-language cases; the poor 3/48 result above
belongs to the Qwen activation model.

## Earlier token-memory experiments

`NeuralTokenMemory` tests a fixed-size parametric alternative. A frozen
Qwen hidden vector is the key; the target is the normalized frozen output
embedding of the demonstrated next token. The stored state is only a learned
linear matrix `W` and an inverse-covariance preconditioner `P`. Each write uses
the squared-error gradient, normalized along the preconditioned key:

```text
k = normalize(hidden)
v = normalize(output_embedding[target_token])
p = P k
error = v - k W
W <- (1 - decay) W + outer(p / (k^T p), error)
P <- P - outer(p, p) / (ridge + k^T p)
```

The implementation computes the error after optional decay. `decay=0` and
`ridge=1e-4` were used in these runs. This is an analytic gradient update,
not a full HOPE implementation or a meta-trained writer. The matrices have
fixed shape 1024 x 1024; FP32 `W` plus FP64 `P` occupies about 12 MiB.
The new-session reader uses the question and its own generated prefix only.
It has no stored examples, text, canonical-question index or target-token
table. A network checkpoint carries the two tensors across sessions.

The [direct Q/A probe](neural-output-memory-v1/README.md) scored **40/40**:
20 distinct questions were written once, queried after a network reload,
then all 20 values were corrected and queried again. This establishes online
parameter storage/correction for demonstrated questions. It does not establish
unrestricted recall from arbitrary statements. Unlike the old latent
self-reconstruction objective, this write has explicit token-level targets.

Automatic statement-to-Q/A translation with temporary paraphrase supervision
scored [6/8 differently worded questions](neural-output-facts-paraphrases-v2/README.md).
Adding memory cosine scores to the frozen model's logits at scale 32 raised
that development result to [7/8](neural-output-logit-connection-v1/README.md);
the base-only connection scored 0/8. Lower scales 8 and 16 scored 1/8 and 5/8.
The scale sweep uses the development questions and is not a held-out result.

An [independent eight-fact probe](neural-output-unseen-facts-v1/README.md)
scored **4/8** with direct readout. Its write trace separates several failures:

- The configuration statement was converted into a question asking which
  repository used a named file. The desired repository-to-file direction
  was never supervised.
- The development-command extraction put its answer in the question; those
  keys were rejected, leaving that fact unwritten.
- The writer omitted the dietary restriction entirely.
- Some generated paraphrases changed the property (log *format* became log
  *location*), training wrong associations. The maintainer query also recalled
  the branch value instead.

These results support token-grounded neural writes and direct output
connections as a useful improvement. They do **not** solve important-fact
selection, paraphrase robustness, unknown-fact abstention, or long-running
capacity. The next experiments should improve and validate the write targets
and train the key representation on independent statement/query pairs; more
sequence length or repetitions of wrong supervision will not fix those errors.
All original Titans/delta behavior and numerical reference tests are unchanged.

## What the experiments establish

The original checkpoint's low loss did not translate into cross-window recall.
Paired examples exposed two training-loss errors: ranking later answer tokens
could see the label under teacher forcing, and ranking the same absolute token
position across paired prompts could compare different answer tokens. The
corrected loss ranks the first differing answer token by answer-token order.
The interrupted checkpoints and their measurements remain in separate experiment
folders.

The corrected broad-fact checkpoint completed 700 MPS optimizer steps over
1,400 training episodes from 14 synthetic user and repository fact categories.
It answered 130/160 validation examples (62/80 complete pairs) and 138/160
held-out test examples (67/80 complete pairs). On the test set, disabling memory
gave 0/160 and resetting it at each window gave 12/160, with no complete pairs
in either ablation. Pair variants have identical final 256 prompt tokens, so
the difference requires information carried from earlier context by the memory
branch. Age and age-correction examples remain weak: 3/12 and 0/10 on test.

The same checkpoint answered only 12/160 examples from three fact categories
excluded from training, with no complete pairs. Replacing familiar held-out
answer values with values absent from training yielded 0/160 on validation and
2/160 on the matched held-out test, with no complete pairs on either. The altered
supporting message remained earlier than the final attention window. This is
strong evidence of selection among learned answer patterns, not reliable
copying of arbitrary new values from fast memory. These are synthetic templates;
they do not measure recall of any important real user or codebase fact.

A direct-context control put each novel supporting fact within Qwen's attention
window, with Titans memory disabled: 38/40 answers were exact. A second control
retrieved one earlier user message by lexical overlap with the question and put
it next to the question; it found the marked supporting message in all 40
validation examples and answered 37/40 exactly (18/20 complete pairs). An
oracle using the marked support span had the same answer score. The three
errors were all time-zone questions, where the decoder sometimes interpreted
a city name as a geographical time zone rather than repeating the stated value.
On the full held-out novel-value test split, lexical retrieval found the marked
support in 160/160 examples and answered 150/160 exactly (71/80 complete pairs),
compared with the neural-memory model's 2/160 and 0/80 pairs on those same items.
The retrieval control is intentionally easy on these templated messages and
uses their original text; its hit rate should not be extrapolated to natural
long-running conversations or codebase search.

The compact Titans fast state can be saved and reloaded at a memory-chunk
boundary. On the first 40 broad validation examples, feeding only the saved
state and the new-session question answered 38/40, versus 2/40 after reset.
A separate two-process CLI smoke check recovered one trained-format repository
build-tool fact. These checks establish persistence of a learned fast state for
trained templates, not arbitrary fact recall across real sessions.
A five-process smoke check with two repository build-tool facts and a correction
exposed interference: it recalled alpha/Bazel before the correction, still said
Bazel after alpha was corrected to Ninja, and said Bazel for beta although beta
was Meson. The scored questions were 1/3 correct. This is one small synthetic
case, but it rules out treating the single-fact success as evidence of reliable
multi-fact storage or updates.

## Architecture implication

The [Titans paper](https://arxiv.org/html/2501.00663) distinguishes contextual
neural memory, which changes with inputs at inference, from fixed task-level
"persistent memory" parameters (sections 3.3 and 4). For a user's facts to
survive a restart, the contextual state must be saved or the source messages
must be replayed. The adapter checkpoint alone cannot hold per-user facts.

The present results favor a hybrid path for practical cross-session memory:
keep source facts in a user- and repository-scoped store, retrieve relevant
evidence at question time, and give the decoder the evidence text. The Titans
branch can be a compact extra signal, but it has not learned open-vocabulary
copying. A future neural-only attempt should train with many held-out value
families, paraphrased statements and queries, multiple facts and corrections,
and an objective that directly rewards copying a value from earlier context.
Select checkpoints on those transfer probes, then evaluate on untouched test
conversations and actual repositories. Increasing sequence length alone will
not address the observed novel-value failure.

## Layer placement and source-text memory follow-up

Qwen3.5's linear Gated DeltaNet layers now accept the same gated Titans branch
as its full-attention layers. A matched placement experiment trained a fresh
layer-10 linear adapter and a fresh layer-11 full-attention adapter for 700 MPS
steps each, with the same seed, data, and optimizer settings. Layer 10 reached
123/160 familiar validation answers and 59/80 complete pairs; layer 11 reached
117/160 and 55/80. Layer 10 scored 1/160 on new answer values and layer 11
scored 0/160, both with no complete pairs. The linear placement modestly
improved trained-template recall, but did not solve open-value copying. The
older layer-11 checkpoint had earlier training and is not the matched control.
See [the placement record](placement-linear10-vs-full11-v1/README.md).

A follow-up [all-layer pilot](placement-all24-vs-full11-pilot200-v1/README.md)
put independent Titans branches on all 18 Gated DeltaNet and 6 full-attention
layers. Scaling each output gate from `-2` to `-5.3` avoided a large startup
loss. At 200 matched steps, all-layer placement scored 6/40 familiar validation
answers and 0/20 complete pairs, versus 13/40 and 2/20 for a fresh layer-11
control. Persisted and reloaded fast states scored 2/20 versus 5/20. Both
models scored 0/40 on novel values. The all-layer adapter weights took 353 MB
and its 200 training steps took 1,191 seconds, versus 14.7 MB and 176 seconds
for layer 11. This limited pilot does not show an accuracy or generalization
benefit from attaching independent memories at every layer; a shared-memory
architecture or different optimization would be a separate test.

The [HOPE paper](https://arxiv.org/html/2512.24695v1) combines self-modifying
projection memories and a continuum memory system with several update rates.
Its long-context model was trained from scratch on roughly 50 billion tokens;
the paper notes that small models' performance can drop without task
fine-tuning. This is not evidence that replacing one frozen-Qwen adapter with
an untrained HOPE-style block would give dependable personal or repository
fact recall. A full HOPE implementation was deferred in favor of measuring
placement and preserving exact source text.

The [shared-layer experiment](placement-shared24-vs-full11-v1/README.md)
tests one Titans module and fast state: all 24 Qwen layers read the
prior-window snapshot, and a learned mixture of their hidden inputs writes
once after the window. This avoids causal leakage from later tokens and keeps
the adapter at 14.7 MB rather than 353 MB for 24 independent adapters. The
fixed write weights remained near uniform after training; token-dependent
cross-layer attention was not tested. After 700 matched steps, the shared
model scored 124/160 familiar test answers and
58/80 complete pairs, versus 117/160 and 49/80 for a fresh layer-11 control.
Serialized-state validation was similar (30/40 versus 31/40). Novel-value test
remained almost entirely unsolved (2/160 versus 0/160, both 0/80 complete
pairs), and a five-process two-repository correction check scored only 1/3.
Cross-layer sharing is useful to test and gives a small familiar-template gain,
but does not fix the intended arbitrary-fact and multi-fact memory behavior.

The new optional episodic text store saves declarative user statements to a
per-scope file and retrieves relevant wording before answering. A later
explicit correction can suppress an older overlapping statement. On novel
values, the final version retrieved the marked support in 160/160 examples,
answered 146/160 validation questions and 149/160 test questions, and solved
70/80 test pairs. On three fact categories excluded from neural training, it
answered 156/160 validation questions and 76/80 pairs. The neural checkpoint
answered 0/160 novel-value validation questions and 2/160 test questions.
An eight-process chat check with two
repository facts, a correction, and two unseen repository values answered all
five scored questions; earlier prompt/retrieval variants answered 3/5 and
4/5. A one-hit retrieval variant found support in only 116/160 novel-value
validation examples, explaining why the final version retains several facts
while suppressing explicit older conflicts. These are synthetic experiments;
automatic retrieval of arbitrary real conversations and codebase observations
still needs a broader evaluation and a way to ingest tool findings.
In a separate three-process check, an ordinary user message automatically
stored an unseen editor and work-hours value. The later answers were exact for
the editor and semantically correct for work hours, where capitalization and a
period prevented exact matching.

## Surprise-delta and repeated test-time examples

The experimental two-timescale delta-rule memory performs a tokenwise gradient
update on key/value prediction error and uses exponential token decay. Its
700-step run processed the first 1,400 of 4,000 generated diverse training
episodes and reached a low training loss, but answered 0/40 held-out novel
values exactly. Normal memory changed predictions between 60% of
counterfactual pair variants;
29/40 outputs were themselves training answer values. Multiplying its read
gates by 4, 8, or 16 still gave 0/40 exact and increasingly malformed text.
The first 40 seen training examples also gave 0/40 exact during generation.
These checks show that the failure is more than a small read gate or a missing
fast-state effect: the decoder does not reliably reconstruct the right value.

In the [repeated-Q/A experiment](repeated-qa-test-time-v1/README.md), each
unseen question/answer pair was written five times into a fresh fast state,
which was saved and reloaded before asking only the question. Exact recall
was 0/40, the same as zero or one presentation. Five Q/A examples visible
directly in a 512-token window with memory disabled gave 40/40. This isolates
the lost information to the compact state path for this prompt/task. Training
with Q/A repetitions, token-level copy supervision, or a different read/write
connection remains untested. An explicit source-text store remains the
reliable implementation for exact user and repository facts in this project.
