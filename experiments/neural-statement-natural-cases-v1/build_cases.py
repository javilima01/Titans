"""Independent manually authored cases; freeze before measuring predictions."""

import json
from pathlib import Path

FACTS = [
    (
        "Please remember that I navigate applications with a screen reader and keyboard only.",
        "screen reader and keyboard only",
        ["How do I interact with applications?", "What accessibility setup do I use?"],
    ),
    (
        "When showing me code, I prefer examples in Rust.",
        "Rust",
        [
            "Which language should your code examples use?",
            "What language do I prefer for sample code?",
        ],
    ),
    (
        "I cannot take calls before 10:00 Europe/Riga.",
        "10:00 Europe/Riga",
        ["How early can you schedule a call with me?", "What is my earliest time for calls?"],
    ),
    (
        "Use CHF as the currency in my expense summaries.",
        "CHF",
        ["Which currency should my expense report use?", "How do I want expenses denominated?"],
    ),
    (
        "My work Git remote is called upstream-work.",
        "upstream-work",
        ["What is the name of my work remote in Git?", "Which Git remote do I use for work?"],
    ),
    (
        "The identity I use when signing commits is Mira Sable.",
        "Mira Sable",
        ["What name do I sign commits with?", "Which identity is used for my commit signatures?"],
    ),
    (
        "Remember: the codename of my side project is Silver Otter.",
        "Silver Otter",
        ["What did I name my side project?", "What is the codename for my side project?"],
    ),
    (
        "I want temperatures expressed in Celsius.",
        "Celsius",
        [
            "What temperature unit do I prefer?",
            "Which scale should you use when telling me the temperature?",
        ],
    ),
    (
        "Code observation in repository LLM: Qwen35Titans is defined in src/llm/modules/qwen.py.",
        "src/llm/modules/qwen.py",
        [
            "Where is the Qwen35Titans wrapper implemented in LLM?",
            "Which LLM source file defines Qwen35Titans?",
        ],
    ),
    (
        "Code observation in repository LLM: TitansMemory lives in src/llm/modules/titans.py.",
        "src/llm/modules/titans.py",
        [
            "Where should I look for TitansMemory in LLM?",
            "Which LLM module implements the Titans memory recurrence?",
        ],
    ),
    (
        "Code observation in repository LLM: TrainingMonitor lives in src/llm/helpers/monitoring.py.",
        "src/llm/helpers/monitoring.py",
        [
            "Where is LLM's TrainingMonitor implemented?",
            "Which file handles LLM's training monitoring?",
        ],
    ),
    (
        "In repository LLM, run all tests with .venv/bin/python -m unittest discover -s tests -v.",
        ".venv/bin/python -m unittest discover -s tests -v",
        ["How do I run the complete LLM test suite?", "What command executes all tests in LLM?"],
    ),
    (
        "In repository LLM, format Python files with .venv/bin/ruff format .",
        ".venv/bin/ruff format .",
        ["How should I format the LLM Python sources?", "What is LLM's Python formatting command?"],
    ),
    (
        "In repository LLM, Python commands must run from the repository root because imports use src.llm.",
        "repository root",
        [
            "Which working directory should I use for LLM Python commands?",
            "Where must I be when launching Python in LLM?",
        ],
    ),
    (
        "The Qwen model cache for repository LLM is .models_cache/Qwen/Qwen3.5-0.8B.",
        ".models_cache/Qwen/Qwen3.5-0.8B",
        [
            "Where does LLM cache its Qwen model?",
            "Which directory holds LLM's downloaded backbone?",
        ],
    ),
    (
        "Code observation in repository LLM: memory adapters use FP32 even with a BF16 backbone.",
        "FP32",
        [
            "What precision are LLM's memory adapters kept in?",
            "Does LLM use BF16 or FP32 for the memory adapters?",
        ],
    ),
    (
        "Code observation in repository aster: the public HTTP routes are registered in app/web/routes_v3.py.",
        "app/web/routes_v3.py",
        [
            "Where does aster register its HTTP endpoints?",
            "Which aster file defines the public routes?",
        ],
    ),
    (
        "In repository aster, launch the development server with pnpm dev.",
        "pnpm dev",
        ["How can I bring up aster locally?", "What command starts aster's development server?"],
    ),
    (
        "Repository aster reads secrets from the ASTER_SECRET_FILE environment variable.",
        "ASTER_SECRET_FILE",
        [
            "Which environment variable points aster to its secrets?",
            "How does aster locate the secrets file?",
        ],
    ),
    (
        "Code observation in repository aster: retry delays use exponential backoff capped at 17 seconds.",
        "17 seconds",
        ["What is the maximum retry delay in aster?", "How long can aster's retry backoff get?"],
    ),
    (
        "Remember: all schema changes in repository aster need approval from the data-platform team.",
        "data-platform team",
        [
            "Who must approve schema changes in aster?",
            "Which team signs off on aster database schema modifications?",
        ],
    ),
    (
        "The generated files in repository aster are under lib/autogen and must never be hand-edited.",
        "lib/autogen",
        [
            "Which aster files should I avoid editing manually?",
            "Where does aster keep its generated files?",
        ],
    ),
    (
        "Repository linden uses the deploy command ./ops/ship --region eu-west-3.",
        "./ops/ship --region eu-west-3",
        ["How do I deploy linden?", "Which command ships linden to its configured region?"],
    ),
    (
        "Repository linden stores its architecture decision records in design/decisions.",
        "design/decisions",
        ["Where are linden's ADRs?", "Where can I read linden's recorded architecture decisions?"],
    ),
]
CORRECTIONS = {
    1: ("Correction: when showing me code, use Zig examples from now on.", "Zig"),
    17: (
        "Correction: in repository aster, launch the development server with just up-local.",
        "just up-local",
    ),
}
NEGATIVES = [
    "Thanks, that was helpful.",
    "How old am I?",
    "If I were using Emacs, would this shortcut work?",
    "Imagine a fictional user whose home city is Quito.",
    "The progress bar has reached 42 percent.",
    "Could you explain what exponential backoff means?",
    "I am waiting for this test run to finish.",
    "Good morning!",
]
UNKNOWN = [
    "What is my phone number?",
    "How old am I?",
    "Which database does aster use?",
    "Who maintains linden?",
    "What is the test command for repository juniper?",
    "What is my favourite editor?",
    "Where are aster's architecture decisions?",
    "What is the deployment command for aster?",
]

folder = Path(__file__).parent
sources = [{"text": row[0], "important": True} for row in FACTS]
sources[5:5] = [{"text": text, "important": False} for text in NEGATIVES]
sources.extend({"text": row[0], "important": True} for row in CORRECTIONS.values())
reads = []
for index, (source, answer, questions) in enumerate(FACTS):
    old_value = None
    if index in CORRECTIONS:
        old_value = answer
        source, answer = CORRECTIONS[index]
    reads.extend(
        {
            "question": question,
            "answer": answer,
            "statement": source,
            "old_value": old_value,
            "category": "correction" if index in CORRECTIONS else "user" if index < 8 else "code",
        }
        for question in questions
    )
reads.extend(
    {"question": question, "answer": "", "statement": "", "category": "unknown"}
    for question in UNKNOWN
)
(folder / "writes.json").write_text(json.dumps({"sources": sources}, indent=2) + "\n")
(folder / "reads.json").write_text(json.dumps({"reads": reads}, indent=2) + "\n")
