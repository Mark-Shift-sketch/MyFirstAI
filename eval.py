import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import Brain
from data.dataset import load_dataset


EVAL_PATH = Path("data/eval_prompts.jsonl")
HISTORY_PATH = Path("data/eval_history.json")
LOW_CONFIDENCE_GATE = 0.40


def _read_eval_rows(path):
    rows = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _load_history(path):
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        if isinstance(payload, list):
            return payload
    except (OSError, json.JSONDecodeError):
        pass
    return []


def _save_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(rows, file_handle, indent=2)


def _week_key(now):
    iso = now.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def run_evaluation():
    eval_rows = _read_eval_rows(EVAL_PATH)
    if not eval_rows:
        print("No evaluation prompts found. Run: python tools/build_eval_set.py")
        return

    brain = Brain(load_dataset())

    total = len(eval_rows)
    correct = 0
    high_bucket = 0
    medium_bucket = 0
    low_bucket = 0

    for row in eval_rows:
        prompt = str(row.get("prompt", "")).strip()
        expected = str(row.get("expected_answer_contains", "")).strip().lower()
        if not prompt or not expected:
            continue

        match = brain.get_match(prompt)
        if not match:
            continue

        predicted = str(match.get("answer", "")).lower()
        score = float(match.get("score", 0.0))

        if score >= 0.72:
            high_bucket += 1
        elif score >= 0.56:
            medium_bucket += 1
        elif score >= LOW_CONFIDENCE_GATE:
            low_bucket += 1

        if score >= LOW_CONFIDENCE_GATE and expected in predicted:
            correct += 1

    accuracy = (correct / total) if total else 0.0
    now = datetime.now()
    record = {
        "week": _week_key(now),
        "timestamp": now.isoformat(timespec="seconds"),
        "total_prompts": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "high_confidence_matches": high_bucket,
        "medium_confidence_matches": medium_bucket,
        "low_confidence_matches": low_bucket,
    }

    history = _load_history(HISTORY_PATH)
    replaced = False
    for idx, item in enumerate(history):
        if item.get("week") == record["week"]:
            history[idx] = record
            replaced = True
            break

    if not replaced:
        history.append(record)

    history = sorted(history, key=lambda item: item.get("week", ""))[-52:]
    _save_history(HISTORY_PATH, history)

    print("Weekly evaluation complete")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    run_evaluation()
