"""Offline feature diagnostic, NOT a reader or a proposed text memory."""

from collections import defaultdict
import json
from pathlib import Path

import torch

from src.llm.helpers.neural_statement_probe import SemanticEncoder, record_probe
from src.llm.modules.neural_statement_memory import NeuralStatementMemory

torch.set_num_threads(4)
root = Path("experiments/neural-statement-stress-cases-v1")
sources = json.loads((root / "writes.json").read_text())["sources"]
reads = [r for r in json.loads((root / "reads.json").read_text())["reads"] if r["answer"]]
encoder = SemanticEncoder()
keys = torch.stack([encoder.encode(s, kind="passage") for s in sources]).double()
queries = torch.stack([encoder.encode(r["question"], kind="query") for r in reads]).double()
indices = torch.tensor([sources.index(r["statement"]) for r in reads])
methods = {"cosine_diagnostic_only": queries @ keys.T}
for bandwidth in (1, 2, 4, 8):
    gram = torch.exp(-bandwidth * torch.cdist(keys, keys).square())
    cross = torch.exp(-bandwidth * torch.cdist(queries, keys).square())
    methods[f"exact_kernel_{bandwidth}_diagnostic_only"] = torch.linalg.solve(
        gram + 1e-4 * torch.eye(len(keys)), cross.T
    ).T
    network = NeuralStatementMemory(feature_width=4096, bandwidth=bandwidth, latent_width=512)
    features = network.features(keys.float()).double()
    query_features = network.features(queries.float()).double()
    gram = features @ features.T
    methods[f"random_features_{bandwidth}"] = torch.linalg.solve(
        gram + 1e-4 * torch.eye(len(keys)), (query_features @ features.T).T
    ).T
methods["linear"] = queries @ keys.T @ torch.linalg.inv(keys @ keys.T + 1e-4 * torch.eye(len(keys)))
results = []
for method, scores in methods.items():
    best = scores.argmax(-1)
    by_category = defaultdict(lambda: [0, 0])
    for i, r in enumerate(reads):
        by_category[r["category"]][0] += int(best[i] == indices[i])
        by_category[r["category"]][1] += 1
    row = {
        "method": method,
        "correct": int((best == indices).sum()),
        "count": len(reads),
        "by_category": dict(by_category),
    }
    results.append(row)
    print(json.dumps(row), flush=True)
record_probe(
    Path("experiments/neural-statement-addressing-v1"),
    {"results": results},
    note="Offline addressing diagnostic with access to source vectors. Cosine and exact-kernel comparisons are diagnostic upper bounds, not admissible network-only readers. Random-feature coefficient ranking is also only a diagnostic, not decoded sentence accuracy.",
)
