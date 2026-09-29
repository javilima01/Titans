"""Create reproducible statement-memory stress cases, with separate write/read files.

These are synthetic probes, not evidence of general natural-language reliability.
No evaluation question or expected answer is input to the writer.
"""

import argparse
import json
from pathlib import Path
import random

REPOSITORIES = ("tamarind", "cobalt", "quasar", "firwood")
PROPERTIES = (
    (
        "entry point",
        "services/{tag}/main.py",
        "Where does {repo} start executing?",
        "Which file is the entry point of {repo}?",
    ),
    (
        "development command",
        "just dev-{tag}",
        "How do I start {repo} locally?",
        "What command launches the {repo} development server?",
    ),
    (
        "test command",
        "uv run pytest tests/{tag} -q",
        "How do I run {repo}'s tests?",
        "Which command checks the {repo} test suite?",
    ),
    (
        "configuration file",
        "config/{tag}.toml",
        "Where is {repo}'s config?",
        "Which file contains configuration for {repo}?",
    ),
    (
        "release branch",
        "release/{tag}",
        "Which branch do we release {repo} from?",
        "What is {repo}'s current release branch?",
    ),
    (
        "maintainer",
        "{tag} Orvex",
        "Who looks after {repo}?",
        "Who maintains the {repo} repository?",
    ),
    (
        "formatter",
        "fmt-{tag}",
        "What formats code in {repo}?",
        "Which formatter should I use for {repo}?",
    ),
    (
        "package manager",
        "pkg-{tag}",
        "What manages {repo}'s packages?",
        "Which package manager does {repo} use?",
    ),
    (
        "artifact directory",
        "out/{tag}/artifacts",
        "Where does {repo} put build artifacts?",
        "Which directory holds artifacts for {repo}?",
    ),
    (
        "documentation directory",
        "docs/{tag}",
        "Where can I find documentation for {repo}?",
        "Which directory contains {repo}'s documentation?",
    ),
    (
        "migration directory",
        "db/{tag}/changes",
        "Where do database migrations go in {repo}?",
        "Which directory contains {repo}'s migrations?",
    ),
    (
        "log format",
        "jsonl-{tag}",
        "How are {repo}'s logs formatted?",
        "What is the log format in {repo}?",
    ),
)
USER_FACTS = (
    (
        "Please remember: my name is Olivane.",
        "Olivane",
        "What should you call me?",
        "What is my name?",
    ),
    ("I am 37 years old.", "37", "How old am I?", "What is my age?"),
    (
        "My preferred editor is Zed.",
        "Zed",
        "Which editor do I prefer?",
        "What editor should you suggest for me?",
    ),
    ("I live in Tartu.", "Tartu", "Which city do I live in?", "Where is my home city?"),
    (
        "My timezone is Europe/Tallinn.",
        "Europe/Tallinn",
        "What time zone do I use?",
        "What is my timezone?",
    ),
    (
        "I prefer explanations with worked examples.",
        "worked examples",
        "How do I like things explained?",
        "What should your explanations include?",
    ),
    (
        "I avoid foods containing hazelnuts.",
        "hazelnuts",
        "Which food ingredient should you avoid recommending to me?",
        "What ingredient do I avoid eating?",
    ),
    (
        "I write dates as DD.MM.YYYY.",
        "DD.MM.YYYY",
        "What date format do I prefer?",
        "How should you format dates for me?",
    ),
    (
        "My working hours are 07:30-15:30.",
        "07:30-15:30",
        "When do I work?",
        "What are my working hours?",
    ),
    (
        "I use the Colemak keyboard layout.",
        "Colemak",
        "Which keyboard layout do I use?",
        "How is my keyboard laid out?",
    ),
    (
        "I prefer distances in kilometres.",
        "kilometres",
        "Which distance unit should you use for me?",
        "How should distances be measured for me?",
    ),
    (
        "I want commit messages written in Spanish.",
        "Spanish",
        "What language should my commit messages use?",
        "In which language do I write commit messages?",
    ),
    (
        "My job title is embedded systems engineer.",
        "embedded systems engineer",
        "What do I do for work?",
        "What is my job title?",
    ),
    (
        "I prefer meetings on Thursdays.",
        "Thursdays",
        "Which day is best for meetings with me?",
        "When do I prefer to have meetings?",
    ),
    (
        "My pronouns are they/them.",
        "they/them",
        "What pronouns should you use for me?",
        "What are my pronouns?",
    ),
    (
        "Remember: my terminal font is Iosevka.",
        "Iosevka",
        "Which font do I use in my terminal?",
        "What is my terminal font?",
    ),
)


def generate(seed: int) -> tuple[dict, dict]:
    rng = random.Random(seed)
    sources, reads, corrections = [], [], []
    for repo in REPOSITORIES:
        for name, value_template, *questions in PROPERTIES:
            tag = "".join(rng.choices("abcdefghjkmnpqrstuvwxyz", k=7))
            value = value_template.format(tag=tag)
            statement = f"The {name} for repository {repo} is {value}."
            sources.append(statement)
            if name == "release branch":
                old_value = value
                value = f"ship/{tag}-v2"
                statement = f"Correction: the {name} for repository {repo} is {value}."
                corrections.append(statement)
            else:
                old_value = None
            reads.extend(
                {
                    "question": question.format(repo=repo),
                    "answer": value,
                    "statement": statement,
                    "category": name,
                    "old_value": old_value,
                }
                for question in questions
            )
    for statement, value, *questions in USER_FACTS:
        sources.append(statement)
        reads.extend(
            {
                "question": question,
                "answer": value,
                "statement": statement,
                "category": "user",
                "old_value": None,
            }
            for question in questions
        )
    rng.shuffle(sources)
    sources.extend(corrections)
    unknown = [
        {
            "question": f"What is the {name} for repository unknown-saffron?",
            "answer": "",
            "statement": "",
            "category": "unknown",
        }
        for name, *_ in PROPERTIES
    ]
    unknown.extend(
        {"question": question, "answer": "", "statement": "", "category": "unknown"}
        for question in (
            "What is my phone number?",
            "What is my dog's name?",
            "Which operating system do I use?",
            "What is my favourite music genre?",
        )
    )
    return {"sources": sources, "seed": seed}, {"reads": reads + unknown, "seed": seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=9041)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    writes, reads = generate(args.seed)
    for name, data in (("writes", writes), ("reads", reads)):
        (args.output / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
