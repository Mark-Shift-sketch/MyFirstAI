import json
from pathlib import Path

LOW_PATH = Path('data/low_confidence.jsonl')
CORPUS = Path('data/corpus.txt')

def main():
    if not LOW_PATH.exists():
        print('No low-confidence file found.')
        return

    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with LOW_PATH.open('r', encoding='utf-8') as fh:
        for line in fh:
            try:
                obj = json.loads(line)
                q = obj.get('query')
                candidate = obj.get('candidate_key')
                if q and candidate:
                    CORPUS.open('a', encoding='utf-8').write(f"{q} ||| {candidate}\n")
                    added += 1
            except Exception:
                continue

    print(f'Appended {added} items to {CORPUS}')

if __name__ == '__main__':
    main()
