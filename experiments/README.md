# Memory experiments

Each folder records the configuration, dataset, checkpoint, and available evaluations.
Re-run `python -m src.llm.helpers.experiment_tracking --checkpoint PATH` after
adding an evaluation report. The checkpoint and dataset artifacts are kept at their
listed paths; compact metrics are copied into each `record.json`.
Older checkpoints predate complete CLI option logging, so some original flags
cannot be recovered from their saved artifacts.
See [research notes](RESEARCH.md) for the benchmark interpretation and next checks.

| Experiment | Steps | Evaluation | EM | Solved pairs | Finding |
| --- | ---: | --- | ---: | ---: | --- |
| [memory-v1](memory-v1/README.md) | n/a | n/a | n/a | n/a | User-reported test: 9/100 exact answers overall; multi-hop 0/37, recall 4/34, update 5/29. |
| [recall-cross-window](recall-cross-window/README.md) | n/a | validation/normal | 0/100 (0.0%) | n/a | Cross-window recall baseline; all three memory modes scored zero on 100 validation examples. |
| [session-profile-smoke-23](session-profile-smoke-23/README.md) | n/a | n/a | n/a | n/a | Single-layer adapter smoke run at layer 23. |
| [session-profile-smoke-11](session-profile-smoke-11/README.md) | n/a | n/a | n/a | n/a | Single-layer adapter smoke run at layer 11. |
| [session-profile-256-v1](session-profile-256-v1/README.md) | n/a | n/a | n/a | n/a | First cross-session profile run using replayed conversation history. |
| [session-profile-gates-smoke](session-profile-gates-smoke/README.md) | n/a | n/a | n/a | n/a | Gate initialization smoke run. |
| [session-profile-batch2-smoke](session-profile-batch2-smoke/README.md) | n/a | n/a | n/a | n/a | Batch-size-two smoke run for cross-session training. |
| [session-profile-256-strong-v1](session-profile-256-strong-v1/README.md) | n/a | validation/normal | 2/24 (8.3%) | n/a | Longer cross-session profile run; early generation evaluation did not establish general recall. |
| [session-profile-256-stage2](session-profile-256-stage2/README.md) | n/a | n/a | n/a | n/a | Second stage of the cross-session profile run. |
| [session-profile-answer-only-v2](session-profile-answer-only-v2/README.md) | n/a | n/a | n/a | n/a | Answer-only objective removed EOS-dominated supervision. |
| [session-profile-answer-only-full-v2](session-profile-answer-only-full-v2/README.md) | n/a | n/a | n/a | n/a | Full answer-only profile training run. |
| [session-paired-overfit-v1](session-paired-overfit-v1/README.md) | n/a | n/a | n/a | n/a | Two-example counterfactual overfit: the adapter learned the distinction after repeated steps. |
| [session-pairs-pilot-v1](session-pairs-pilot-v1/README.md) | n/a | n/a | n/a | n/a | First paired-dataset pilot. |
| [session-pairs-mps-smoke-v1](session-pairs-mps-smoke-v1/README.md) | n/a | n/a | n/a | n/a | Apple MPS training smoke run. |
| [session-pairs-pilot-mps-v1](session-pairs-pilot-mps-v1/README.md) | n/a | validation/normal | 0/24 (0.0%) | 0/12 | One layer, 50 pairs, one pass: zero fully correct validation pairs. |
| [session-pairs-repeat-mps-v1](session-pairs-repeat-mps-v1/README.md) | n/a | n/a | n/a | n/a | Four more epochs on the same 50 pairs: zero fully correct held-out pairs. |
| [session-pairs-aligned-pilot-v1](session-pairs-aligned-pilot-v1/README.md) | n/a | n/a | n/a | n/a | Aligned query/key initialization improved a diagnostic margin but solved no held-out pairs. |
| [session-pairs-delta-pilot-v1](session-pairs-delta-pilot-v1/README.md) | n/a | n/a | n/a | n/a | Fast-weight-only delta read lowered CE but did not solve held-out pairs. |
| [session-pairs-clip10-pilot-v1](session-pairs-clip10-pilot-v1/README.md) | n/a | n/a | n/a | n/a | Inner gradient norm cap of 10 destabilized validation loss. |
| [session-pairs-two-layer-pilot-v1](session-pairs-two-layer-pilot-v1/README.md) | n/a | n/a | n/a | n/a | Two-layer adapter did not improve held-out paired recall in the pilot. |
| [session-pairs-full-v1](session-pairs-full-v1/README.md) | n/a | validation/normal | 1/24 (4.2%) | 0/12 | Full 600-pair answer-only training learned common answer tokens but did not use memory reliably. |
| [session-pairs-contrastive-pilot-v1](session-pairs-contrastive-pilot-v1/README.md) | n/a | n/a | n/a | n/a | First paired ranking-loss pilot improved the answer margin; held-out pairs remained unsolved. |
| [session-pairs-contrastive-full-v1](session-pairs-contrastive-full-v1/README.md) | n/a | validation/normal | 27/160 (16.9%) | 10/80 | Paired ranking objective yields a clear memory effect on held-out session facts. |
| [session-age-pairs-pilot-v1](session-age-pairs-pilot-v1/README.md) | n/a | validation/normal | 1/24 (4.2%) | 0/12 | Age-focused pilot with all differing answer tokens and EOS supervision: changed answers, but zero fully correct pairs. |
| [session-age-pairs-full-v1](session-age-pairs-full-v1/README.md) | n/a | validation/normal | 6/24 (25.0%) | 1/12 | Interrupted after 249 steps when a teacher-forcing shortcut was found in the later-digit ranking loss; partial age checkpoint retained for diagnosis. |
| [session-broad-pairs-full-v1](session-broad-pairs-full-v1/README.md) | n/a | validation/normal | 23/40 (57.5%) | 8/20 | Interrupted after 154 steps: 150/700 pairs had unequal prompt token lengths, while the old ranking loss compared equal absolute positions. This checkpoint is diagnostic only. |
| [broad-heldout-test-v1](broad-heldout-test-v1/README.md) | n/a | test/normal | 138/160 (86.2%) | 67/80 | Held-out 14-category test after checkpoint selection on validation. |
| [broad-novel-values-v1](broad-novel-values-v1/README.md) | n/a | validation/normal | 0/160 (0.0%) | 0/80 | Replaced held-out answers with values absent from the training pools while keeping the same fact formats and questions. |
| [broad-unseen-categories-v1](broad-unseen-categories-v1/README.md) | n/a | validation/normal | 12/160 (7.5%) | 0/80 | Held out work hours, deployment region, and release branch from training to test new fact types. |
| [novel-neural-test-v1](novel-neural-test-v1/README.md) | n/a | test/normal | 2/160 (1.2%) | 0/80 | Matched 160-example novel-value test for trained neural memory; all answer values excluded from training pools. |
| [novel-retrieval-test-v1](novel-retrieval-test-v1/README.md) | n/a | test/lexical | 150/160 (93.8%) | 71/80 | Lexical text-retrieval control on all 160 held-out novel-value test episodes. |
| [novel-retrieval-validation-40-v1](novel-retrieval-validation-40-v1/README.md) | n/a | validation/lexical | 37/40 (92.5%) | 18/20 | Lexical retrieval and oracle-support controls on 40 held-out novel-value validation episodes. |
| [novel-values-direct-context-v1](novel-values-direct-context-v1/README.md) | n/a | validation/disabled | 38/40 (95.0%) | 18/20 | Control: moved the fact into one 512-token attention window and disabled neural memory to test frozen Qwen copying. |
| [serialized-state-validation-v1](serialized-state-validation-v1/README.md) | n/a | validation/state_only | 38/40 (95.0%) | 18/20 | Controlled cross-session test: save and reload fast memory after prior sessions; ask with only the new-session prompt. |
| [session-broad-pairs-aligned-v1](session-broad-pairs-aligned-v1/README.md) | 700 | validation/normal | 130/160 (81.2%) | 62/80 | Strong single-fact recall on trained synthetic templates; novel values, unseen categories, and multi-fact corrections remain unreliable. |
| [session-state-cli-smoke-v1](session-state-cli-smoke-v1/README.md) | n/a | n/a | n/a | n/a | Second process answered Bazel correctly from saved state; identical question without saved state answered Zed. The first process replied 3; state file was 4.0 MB. |
| [session-state-multifact-smoke-v1](session-state-multifact-smoke-v1/README.md) | n/a | n/a | n/a | n/a | Before correction: alpha/Bazel correct. After correction: alpha/Ninja was answered Bazel; beta/Meson was also answered Bazel. Scored 1/3 questions; state file 4.0 MB. |
