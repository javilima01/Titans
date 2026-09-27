# Memory research notes

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
