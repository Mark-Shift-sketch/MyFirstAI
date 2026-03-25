import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import load_dataset


EVAL_PATH = Path("data/eval_prompts.jsonl")
TARGET_COUNT = 240


def _clean(text):
    return re.sub(r"\s+", " ", str(text or "").strip())


def _expected_snippet(answer):
    words = re.findall(r"[a-z0-9']+", answer.lower())
    if not words:
        return ""
    snippet = " ".join(words[: min(5, len(words))])
    return snippet


def _variations(question):
    q = _clean(question).lower().strip(" ?")
    if not q:
        return []

    variants = [
        q,
        f"{q}?",
        f"please {q}",
        f"can you tell me {q}",
        f"could you explain {q}",
        f"jarvis {q}",
        f"i want to know {q}",
        f"answer this: {q}",
        f"can you answer {q}",
        f"quick question {q}",
    ]

    if q.startswith("what is "):
        tail = q.replace("what is ", "", 1)
        variants.append(f"whats {tail}")
    if q.startswith("who is "):
        tail = q.replace("who is ", "", 1)
        variants.append(f"tell me about {tail}")

    deduped = []
    seen = set()
    for item in variants:
        cleaned = _clean(item)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped


def build_eval_set(target_count=TARGET_COUNT):
    dataset = load_dataset()
    rows = []

    for item in dataset:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        expected = _expected_snippet(answer)
        if not question or not answer or not expected:
            continue

        for prompt in _variations(question):
            rows.append(
                {
                    "prompt": prompt,
                    "expected_answer_contains": expected,
                    "source_question": question,
                }
            )

    deduped_rows = []
    seen_prompts = set()
    for row in rows:
        prompt = row["prompt"].lower()
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        deduped_rows.append(row)

    if len(deduped_rows) > target_count:
        deduped_rows = deduped_rows[:target_count]

    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_PATH.open("w", encoding="utf-8") as file_handle:
        for row in deduped_rows:
            file_handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    return len(deduped_rows)


if __name__ == "__main__":
    total = build_eval_set()
    print(f"Wrote {total} evaluation prompts to {EVAL_PATH}")
