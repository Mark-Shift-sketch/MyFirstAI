# Supervised Fine-Tuning Dataset Extraction
# Format: instruction, response, context (if available)
# Source: data/corpus.txt (auto-logged by _record_interaction_for_training)

import json
from pathlib import Path

CORPUS_PATH = Path("data/corpus.txt")
OUTPUT_PATH = Path("data/fine_tune_dataset.jsonl")

examples = []

with CORPUS_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or "|||" not in line:
            continue
        user, assistant = line.split("|||", 1)
        user = user.strip()
        assistant = assistant.strip()
        if user and assistant:
            examples.append({
                "instruction": user,
                "response": assistant
            })

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Wrote {len(examples)} examples to {OUTPUT_PATH}")
