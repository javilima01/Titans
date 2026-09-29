"""Natural-statement writes and independent query wording for neural memory.

Generated Q/A supervision exists only during each write. Reads call the network
with the new question alone; reports never enter the prediction path.
"""

import argparse
from contextlib import suppress
import json
from pathlib import Path
import re
import time

from safetensors.torch import load_file, save_file
import torch

from src.llm.helpers.neural_memory_probe import NeuralMemoryProbe
from src.llm.modules.neural_token_memory import NeuralTokenMemory
from src.llm.modules.qwen import Qwen35Wrapper

SOURCES = (
    "Please remember: My name is Ilaria.",
    "I prefer explanations with worked examples.",
    "In repository kestrel, run tests with uv run pytest tests/api -q.",
    "The entry point for repository kestrel is services/gateway/start.py.",
    "The build tool for repository fern is Meson.",
    "For repository kestrel, use the build tool Bazel.",
    "Correction: the build tool for repository kestrel is Ninja.",
    "My timezone is Pacific/Chatham.",
    "The migration policy for kestrel is append-only SQL files under db/changes.",
)
READS = (
    ("What should you call me?", "Ilaria"),
    ("How do I like things explained?", "explanations with worked examples"),
    ("Which command runs kestrel's API tests?", "uv run pytest tests/api -q"),
    ("Where does the kestrel service start?", "services/gateway/start.py"),
    ("What builds fern?", "Meson"),
    ("What builds kestrel now?", "Ninja"),
    ("What time zone do I use?", "Pacific/Chatham"),
    (
        "Where do kestrel's database changes go, and can we edit old migrations?",
        "append-only SQL files under db/changes",
    ),
)
EXTRACT = """Convert the user's factual statements into questions and answers for later recall.
Preserve the named subject, repository and property in each question. Use first person for user facts.
Answers must be exact substrings of the user's text. Do not make assumptions.
Return ONLY JSON: [{"question":"...", "answer":"..."}].
Return [] for questions, hypotheticals, small talk or requests with no stated fact.
"""
EXAMPLES = (
    (
        "Remember: I use the editor Helix.",
        '[{"question":"Which editor do I use?","answer":"Helix"}]',
    ),
    (
        "The deployment file for repo amber is ops/deploy.sh.",
        '[{"question":"What is the deployment file for repo amber?","answer":"ops/deploy.sh"}]',
    ),
    ("If my name were Kai, what would you call me?", "[]"),
)
PARAPHRASE_EXAMPLES = (
    (
        "Where are the logs for repository cedar?",
        '["Where does cedar keep its log files?", "Which directory contains the cedar logs?", "What is the logs location for cedar?", "Where can I find cedar logs?"]',
    ),
)


@torch.no_grad()
def complete(model, instruction, content, *, examples=(), max_new_tokens=384):
    messages = [{"role": "system", "content": instruction}]
    for user, assistant in examples:
        messages.extend(
            [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        )
    messages.append({"role": "user", "content": content})
    text = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = model.tokenizer(text, return_tensors="pt").to(model.device)
    output = model.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return model.tokenizer.decode(
        output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()


def parse(text):
    return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paraphrases", action="store_true")
    parser.add_argument(
        "--cases", type=Path, help="Independent source statements and read questions"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    torch.set_num_threads(4)
    model = Qwen35Wrapper(device="mps")
    probe = NeuralMemoryProbe(model)
    case_data = (
        json.loads(args.cases.read_text()) if args.cases else {"sources": SOURCES, "reads": READS}
    )
    writes, audit_questions = [], []
    started = time.monotonic()
    for source in case_data["sources"]:
        raw = complete(model, EXTRACT, source, examples=EXAMPLES)
        try:
            facts = parse(raw)
        except ValueError:
            facts = []
        accepted = []
        for fact in facts:
            if (
                not isinstance(fact, dict)
                or not isinstance(fact.get("answer"), str)
                or not isinstance(fact.get("question"), str)
                or not fact["answer"]
                or fact["answer"] not in source
            ):
                continue
            question = fact["question"].replace("your ", "my ").replace("the user's ", "my ")
            questions = [question]
            variants = None
            if args.paraphrases:
                variants = complete(
                    model,
                    "Write four differently worded questions asking for exactly the same information. Keep the same person and repository. Do not answer the question. Return ONLY a JSON array of strings.",
                    question,
                    examples=PARAPHRASE_EXAMPLES,
                )
                with suppress(ValueError):
                    questions.extend(
                        value
                        for value in parse(variants)
                        if isinstance(value, str) and value.endswith("?")
                    )
            questions = list(
                dict.fromkeys(
                    question
                    for question in questions
                    if fact["answer"].casefold() not in question.casefold()
                )
            )
            for question in questions:
                probe.teach(question, fact["answer"])
            accepted.append(
                {"questions": questions, "answer": fact["answer"], "raw_paraphrases": variants}
            )
            audit_questions.extend(questions)
        row = {"source": source, "raw": raw, "accepted": accepted}
        writes.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    path = args.output.parent / "network.safetensors"
    save_file(probe.memory.state_dict(), path)
    probe.memory = NeuralTokenMemory(probe.memory.weight.shape[0])
    probe.memory.load_state_dict(load_file(path))
    results = []
    for question, expected in case_data["reads"]:
        prediction, confidence = probe.answer(question)
        row = {
            "question": question,
            "expected": expected,
            "prediction": prediction,
            "confidence": confidence,
            "correct": prediction == expected,
            "question_seen_during_write": question in audit_questions,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "paraphrases": args.paraphrases,
                "cases": str(args.cases) if args.cases else "development",
                "writes": writes,
                "reads": results,
                "correct": sum(row["correct"] for row in results),
                "count": len(results),
                "network_bytes": path.stat().st_size,
                "elapsed_seconds": time.monotonic() - started,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
