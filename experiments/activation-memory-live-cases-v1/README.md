# activation-memory-live-cases-v1

Take validation records in original order, select rows [::2][:20] to avoid conflicting counterfactual variants, put source statements into writes.json and question/answer pairs into reads.json. Twenty coexisting facts. Evaluated by separate live-write/live-read processes without reading the feature cache.

Source: `experiments/activation-memory-features-v1/cases.json`

Questions and expected answers in reads.json are excluded from writer inputs. Case files are benchmark/audit artifacts.
