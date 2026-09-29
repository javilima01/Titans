"""Write whole statements into dense network weights; read using new questions.

E5 supplies semantic features, not a retrieval index. No factual text is passed
to the reader, which uses only the encoder and the fixed-size fast network.
"""

import argparse
from contextlib import suppress
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shlex
import sys
import time

from safetensors.torch import load_file, save_file
import torch
from torch.nn import functional
from transformers import AutoModel, AutoTokenizer

from src.llm.helpers.neural_fact_probe import complete, parse
from src.llm.modules.neural_statement_memory import NeuralStatementMemory
from src.llm.modules.qwen import Qwen35Wrapper

QUESTION_INSTRUCTION = """Write three short, differently worded questions whose answer is given by the supplied statement.
Preserve the person and named repository. Ask for the stated property, leaving its value out of the question.
For corrections, ask about the current value. Output ONLY a JSON list of strings. Do not answer them."""
QUESTION_EXAMPLES = (
    (
        "My editor is Helix.",
        '["Which editor do I use?", "What is my preferred editor?", "What editor should you recommend to me?"]',
    ),
    (
        "The deployment file for repo amber is ops/deploy.sh.",
        '["Where is amber\'s deployment file?", "Which file deploys repository amber?", "Where should I look to deploy amber?"]',
    ),
    (
        "I avoid shellfish.",
        '["Which food do I avoid?", "What food should you not recommend to me?", "What is my dietary restriction?"]',
    ),
)


def write_questions(model, source: str, *, normalize: bool = False) -> tuple[str, list[str]]:
    if normalize:
        source = re.sub(
            r"^(?:please\s+)?(?:remember|correction|update|note):\s*",
            "",
            source,
            flags=re.IGNORECASE,
        )
    raw = complete(
        model, QUESTION_INSTRUCTION, source, examples=QUESTION_EXAMPLES, max_new_tokens=192
    )
    questions = []
    with suppress(ValueError, TypeError):
        values = parse(raw)
        if isinstance(values, list):
            questions = list(
                dict.fromkeys(
                    value for value in values if isinstance(value, str) and value.endswith("?")
                )
            )[:3]
    if normalize:
        match = re.search(r"\brepo(?:sitory)?\s+([\w-]+)", source, flags=re.IGNORECASE)
        if match:
            repository = match.group(1)
            questions = [
                question
                if repository.casefold() in question.casefold()
                else f"For repository {repository}, {question[0].lower()}{question[1:]}"
                for question in questions
            ]
    return raw, questions


def important_statement(model, source: str) -> tuple[bool, str]:
    raw = complete(
        model,
        "Does this message state a durable fact about the user or their codebase that would help in a future session, or explicitly ask to remember a stated fact? Reply only yes or no. Questions, hypotheticals, temporary progress, greetings and quoted fictional examples are not durable facts.",
        source,
        examples=(
            ("I prefer short explanations.", "yes"),
            ("Code observation: tests run using just check.", "yes"),
            ("Thanks, that worked.", "no"),
            ("Suppose my editor were Vim; what would you suggest?", "no"),
        ),
        max_new_tokens=4,
    )
    return raw.strip().lower().rstrip(".!") == "yes", raw


def answers_question(model, question: str, statement: str) -> tuple[bool, str]:
    raw = complete(
        model,
        "Does the single remembered fact directly answer the question? Check the exact person or repository AND the requested property. A related topic is insufficient. Do not use outside knowledge. Reply only yes or no.",
        f"Question: {question}\nFact: {statement}",
        examples=(
            ("Question: Where is amber's config?\nFact: amber's tests run with just check.", "no"),
            (
                "Question: Where is amber's config?\nFact: amber loads settings from cfg/dev.toml.",
                "yes",
            ),
            ("Question: How old am I?\nFact: My name is Lea.", "no"),
            ("Question: Who maintains oak?\nFact: Mira maintains ash.", "no"),
        ),
        max_new_tokens=4,
    )
    return raw.strip().lower().rstrip(".!") == "yes", raw


class SemanticEncoder:
    """Frozen query/passage encoder following the E5 model card."""

    def __init__(self, *, device: str = "mps"):
        path = ".models_cache/intfloat/e5-small-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def encode(self, text: str, *, kind: str) -> torch.Tensor:
        inputs = self.tokenizer(f"{kind}: {text}", return_tensors="pt").to(self.device)
        if inputs.input_ids.shape[1] > 512:
            msg = "Encoder input exceeds 512 tokens"
            raise ValueError(msg)
        hidden = self.model(**inputs).last_hidden_state
        mask = inputs.attention_mask[..., None].bool()
        pooled = hidden.masked_fill(~mask, 0).sum(1) / mask.sum(1)
        return functional.normalize(pooled, dim=-1)[0].cpu().float()


def record_probe(folder: Path, report: dict, *, note: str) -> None:
    """Keep each experiment self-contained, including failures and its command."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    command = shlex.join(sys.orig_argv)
    record = {
        "id": folder.name,
        "kind": "analysis",
        "checkpoint": str(folder / "network.safetensors")
        if (folder / "network.safetensors").exists()
        else report.get("checkpoint") or report.get("config", {}).get("state"),
        "source": report.get("cases")
        or report.get("data")
        or report.get("cache")
        or report.get("config", {}).get("cases"),
        "training": {},
        "reports": {},
        "note": note,
        "result": note,
        "command": command,
        "recorded_at": datetime.now(UTC).isoformat(),
        "status": "experimental",
    }
    rows = report.get("results", report.get("reads", []))
    metric = "correct" if rows and "correct" in rows[0] else "exact_statement"
    scored = [row for row in rows if isinstance(row, dict) and isinstance(row.get(metric), bool)]
    if scored:
        known = [row for row in scored if row.get("answer", row.get("expected")) != ""]
        unknown = [row for row in scored if row.get("answer", row.get("expected")) == ""]
        evaluations = {}
        if known:
            evaluations["normal"] = {
                "metrics": {
                    "count": len(known),
                    "exact_match": sum(row[metric] for row in known) / len(known),
                    "match_target": "answer" if metric == "correct" else "complete_statement",
                }
            }
        if unknown:
            evaluations["unknown"] = {
                "metrics": {
                    "count": len(unknown),
                    "empty_output_rate": sum(not row["prediction"].strip() for row in unknown)
                    / len(unknown),
                }
            }
        record["reports"] = {
            "results.json": {
                "split": report.get("split", "development"),
                "evaluation_kind": "neural_parameters_only",
                "evaluations": evaluations,
            }
        }
    (folder / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    (folder / "README.md").write_text(
        f"# {folder.name}\n\n{note}\n\n```sh\n{command}\n```\n\n"
        "See results.json for every prediction. Source text in this report is an audit artifact; "
        "the reader accepts only neural parameters and questions. See results.json for the architecture. "
        "This is a bounded research probe, not an automatic chat memory.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", type=int, default=1024)
    parser.add_argument("--bandwidth", type=float, default=4.0)
    parser.add_argument("--code-width", type=int, default=64)
    parser.add_argument("--latent-width", type=int, default=0)
    parser.add_argument("--latent-sparsity", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--write-only", action="store_true")
    parser.add_argument("--write-questions", action="store_true")
    parser.add_argument("--normalize-write-questions", action="store_true")
    parser.add_argument("--select-important", action="store_true")
    parser.add_argument("--verify-read", action="store_true")
    parser.add_argument("--read-kind", choices=("query", "passage"), default="query")
    parser.add_argument(
        "--state", type=Path, help="Read from a network checkpoint in a fresh process"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new experiment folder")
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    encoder = SemanticEncoder(device=args.device)
    writer = (
        Qwen35Wrapper(device=args.device)
        if ((args.write_questions or args.select_important) and not args.state) or args.verify_read
        else None
    )
    config = {
        "feature_width": args.features,
        "bandwidth": args.bandwidth,
        "code_width": args.code_width,
        "latent_width": args.latent_width,
        "latent_sparsity": args.latent_sparsity,
    }
    if args.state:
        config = json.loads((args.state.parent / "network_config.json").read_text())
    memory = NeuralStatementMemory(**config)
    cases = json.loads(args.cases.read_text())
    started = time.monotonic()
    writes = []
    if args.state:
        memory.load_state_dict(load_file(args.state))
    for source in () if args.state else cases["sources"]:
        expected_selection = source.get("important") if isinstance(source, dict) else None
        source = source["text"] if isinstance(source, dict) else source
        accepted, selection_raw = (
            important_statement(writer, source) if args.select_important else (True, None)
        )
        if not accepted:
            writes.append(
                {
                    "source": source,
                    "accepted": False,
                    "expected_selection": expected_selection,
                    "selection_raw": selection_raw,
                }
            )
            continue
        raw, questions = (
            write_questions(writer, source, normalize=args.normalize_write_questions)
            if args.write_questions
            else (None, [])
        )
        hidden = [encoder.encode(source, kind="passage")]
        hidden.extend(encoder.encode(question, kind="query") for question in questions)
        result = memory.write(torch.stack(hidden), source)
        row = {
            "source": source,
            "questions": questions,
            "raw_questions": raw,
            "accepted": True,
            "expected_selection": expected_selection,
            "selection_raw": selection_raw,
            **result,
        }
        writes.append(row)
        if writer:
            print(json.dumps(row), flush=True)
    state = args.state or args.output / "network.safetensors"
    if not args.state:
        save_file(memory.state_dict(), state)
        (args.output / "network_config.json").write_text(json.dumps(config, indent=2) + "\n")
        memory = NeuralStatementMemory(**config)
        memory.load_state_dict(load_file(state))
    reads = []
    for case in () if args.write_only else cases["reads"]:
        if isinstance(case, dict):
            question, expected = case["question"], case["answer"]
            expected_statement = case.get("statement")
        else:
            question, expected = case
            expected_statement = None
        prediction, confidence = memory.recall(encoder.encode(question, kind=args.read_kind))
        raw_prediction = prediction
        verification_raw = None
        if args.verify_read:
            verified, verification_raw = answers_question(writer, question, prediction)
            prediction = prediction if verified else ""
        row = {
            "question": question,
            "expected": expected,
            "prediction": prediction,
            "raw_prediction": raw_prediction,
            "verification_raw": verification_raw,
            "confidence": confidence,
            "contains_answer": expected in prediction if expected else not prediction,
            "expected_statement": expected_statement,
            "exact_statement": prediction == expected_statement
            if expected_statement is not None
            else None,
            "category": case.get("category", "unspecified")
            if isinstance(case, dict)
            else "unspecified",
            "stale": bool(case.get("old_value") and case["old_value"] in prediction)
            if isinstance(case, dict)
            else False,
        }
        reads.append(row)
        print(
            json.dumps({key: value for key, value in row.items() if key != "confidence"}),
            flush=True,
        )
    correct = sum(row["contains_answer"] for row in reads)
    report = {
        "config": vars(args)
        | {
            "cases": str(args.cases),
            "output": str(args.output),
            "state": str(args.state) if args.state else None,
        },
        "writes": writes,
        "reads": reads,
        "contains_answer": correct,
        "count": len(reads),
        "exact_statement": sum(row["exact_statement"] is True for row in reads),
        "network_bytes": state.stat().st_size,
        "elapsed_seconds": time.monotonic() - started,
    }
    record_probe(
        args.output,
        report,
        note=f"Complete-statement neural regression: {correct}/{len(reads)} contain the requested value. "
        "Contains-answer is lenient; inspect sentence corruption, wrong subjects and stale values separately.",
    )


if __name__ == "__main__":
    main()
