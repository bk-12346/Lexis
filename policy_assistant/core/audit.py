import json
import os
from datetime import datetime


AUDIT_LOG_PATH = "policy_assistant/audit_log.jsonl"


def log_query(
    user: str,
    question: str,
    answer: str,
    sources: list[dict],
    confidence: float,
    filters: dict = None
) -> None:
    """
    Append a query to the audit log.

    In production this would write to a database.
    We use JSONL (one JSON object per line) — easy to parse,
    easy to stream, works with log aggregators like Splunk.

    This is non-negotiable for government clients — every query
    must be traceable.
    """
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user": user,
        "question": question,
        "answer": answer,
        "confidence": confidence,
        "filters_applied": filters or {},
        "sources_cited": [
            f"{s['title']} | Page {s['page']}"
            for s in sources
        ],
        "source_count": len(sources)
    }

    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def get_recent_queries(n: int = 10) -> list[dict]:
    """Return the n most recent audit log entries."""
    if not os.path.exists(AUDIT_LOG_PATH):
        return []

    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entries = [json.loads(line) for line in lines if line.strip()]
    return entries[-n:]