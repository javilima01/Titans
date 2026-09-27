"""Persistent source-text memory for facts that must survive process restarts.

This store keeps the user's original assertions and retrieves relevant evidence
at answer time. It deliberately does not compress answer values into model
weights. One file is one memory scope (for example, one user and project).
"""

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import tempfile

STOPWORDS = {
    "a",
    "am",
    "and",
    "are",
    "do",
    "does",
    "for",
    "how",
    "i",
    "in",
    "is",
    "me",
    "my",
    "of",
    "on",
    "please",
    "reply",
    "should",
    "the",
    "to",
    "use",
    "what",
    "when",
    "where",
    "which",
    "with",
    "you",
}
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|\n+")
CORRECTION = re.compile(r"\b(correction|actually|instead)\b", re.IGNORECASE)


def _terms(text: str) -> set[str]:
    return {
        word[:5]
        for word in re.findall(r"\w+", text.casefold())
        if len(word) >= 3 and word not in STOPWORDS
    }


@dataclass(frozen=True)
class EvidenceRecord:
    ordinal: int
    text: str


def evidence_message(records: list[EvidenceRecord]) -> str:
    """Format exact prior wording, keeping later corrections last."""
    if not records:
        return ""
    notes = "\n".join(f"[{record.ordinal + 1}] {record.text}" for record in records)
    return (
        "Earlier user statements, oldest to newest. Later corrections override "
        f"earlier statements about the same subject.\n{notes}"
    )


def prompt_with_evidence(question: str, records: list[EvidenceRecord]) -> str:
    """Build a single-turn prompt for callers that cannot pass chat messages."""
    if not records:
        return question
    return f"{evidence_message(records)}\n\nCurrent message: {question}"


class EpisodicMemory:
    """Append user statements, retrieve matching evidence, and save atomically."""

    def __init__(self, records: list[EvidenceRecord] | None = None):
        self.records = list(records or [])

    def remember(self, message: str) -> int:
        """Store declarative snippets verbatim; return the number added."""
        added = 0
        for part in SENTENCE_BOUNDARY.split(message.strip()):
            statement = part.strip()
            if not statement or statement.endswith("?"):
                continue
            if self.records and self.records[-1].text == statement:
                continue
            self.records.append(EvidenceRecord(len(self.records), statement))
            added += 1
        return added

    def retrieve(
        self, query: str, *, limit: int = 3, max_chars: int = 1000
    ) -> list[EvidenceRecord]:
        """Rank lexical overlap with IDF and recency; present hits in time order."""
        if limit < 1 or max_chars < 1:
            msg = "limit and max_chars must be positive"
            raise ValueError(msg)
        query_terms = _terms(query)
        if not query_terms or not self.records:
            return []
        documents = [_terms(record.text) for record in self.records]
        frequency = Counter(term for document in documents for term in document)
        scores = [
            sum(
                math.log1p((len(documents) + 1) / (frequency[term] + 1))
                for term in query_terms & document
            )
            for document in documents
        ]
        corrections = [
            index
            for index, record in enumerate(self.records)
            if CORRECTION.search(record.text) and scores[index] > 0
        ]
        scored = []
        for index, (record, document) in enumerate(zip(self.records, documents, strict=True)):
            score = scores[index]
            if score <= 0:
                continue
            superseded = any(
                later > index
                and scores[later] >= score
                and len(document & documents[later]) / len(document | documents[later]) >= 0.4
                for later in corrections
            )
            if not superseded:
                scored.append((score, record.ordinal, record))
        selected = sorted(scored, reverse=True)[:limit]
        total = 0
        kept = []
        for _, _, record in selected:
            if total + len(record.text) > max_chars:
                continue
            kept.append(record)
            total += len(record.text)
        return sorted(kept, key=lambda record: record.ordinal)

    @classmethod
    def load(cls, path: str | Path) -> EpisodicMemory:
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1:
            msg = f"Unsupported episodic memory file: {path}"
            raise ValueError(msg)
        raw_records = data.get("records")
        if not isinstance(raw_records, list):
            msg = f"Invalid episodic memory records: {path}"
            raise ValueError(msg)
        records = []
        for index, raw in enumerate(raw_records):
            if (
                not isinstance(raw, dict)
                or raw.get("ordinal") != index
                or not isinstance(raw.get("text"), str)
            ):
                msg = f"Invalid episodic memory record {index}: {path}"
                raise ValueError(msg)
            records.append(EvidenceRecord(index, raw["text"]))
        return cls(records)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "records": [asdict(record) for record in self.records]}
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            try:
                temporary.chmod(0o600)
                json.dump(data, handle, ensure_ascii=False)
                handle.write("\n")
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(path)
