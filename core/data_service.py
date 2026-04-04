from db.connection import SessionLocal
from db.models import DatasetEntry

def load_dataset():
    db = SessionLocal()
    try:
        entries = db.query(DatasetEntry).all()
        # Returns list of dicts with question/answer
        return [{'question': e.question, 'answer': e.answer} for e in entries]
    except Exception as e:
        print(f'Database error: {e}, returning empty dataset')
        return []
    finally:
        db.close()
