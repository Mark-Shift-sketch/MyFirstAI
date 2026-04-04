import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Brain:
    def __init__(self, dataset, model_name="all-MiniLM-L6-v2"):
        self.entries = []
        for item in dataset:
            if isinstance(item, dict):
                question = str(item.get("question", "")).strip().lower()
                answer = str(item.get("answer", "")).strip()
                metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
            else:
                question, answer = item
                question = str(question).strip().lower()
                answer = str(answer).strip()
                metadata = {}

            if question and answer:
                self.entries.append(
                    {
                        "question": question,
                        "answer": answer,
                        "metadata": metadata,
                    }
                )

        if not self.entries:
            self.entries = [
                {
                    "question": "hello",
                    "answer": "Hello. How can I assist you?",
                    "metadata": {"source": "fallback", "confidence": 1.0},
                }
            ]

        self.questions = [entry["question"] for entry in self.entries]
        self.embedder = SentenceTransformer(model_name)
        matrix = self.embedder.encode(
            self.questions,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.embeddings = np.asarray(matrix, dtype="float32")

        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def get_match(self, user_input):
        prompt = str(user_input or "").strip().lower()
        if not prompt:
            return None

        query = self.embedder.encode([prompt], convert_to_numpy=True, normalize_embeddings=True)
        query = np.asarray(query, dtype="float32")

        scores, indices = self.index.search(query, 1)
        index = int(indices[0][0])
        score = float(scores[0][0])

        if index < 0 or index >= len(self.entries):
            return None

        entry = self.entries[index]
        return {
            "question": entry["question"],
            "answer": entry["answer"],
            "metadata": entry.get("metadata", {}),
            "score": score,
        }

    def get_top_matches(self, user_input, k=3, min_score=0.0):
        prompt = str(user_input or "").strip().lower()
        if not prompt:
            return []

        top_k = max(1, min(int(k), len(self.entries)))
        query = self.embedder.encode([prompt], convert_to_numpy=True, normalize_embeddings=True)
        query = np.asarray(query, dtype="float32")

        scores, indices = self.index.search(query, top_k)
        matches = []
        for score, index in zip(scores[0], indices[0]):
            idx = int(index)
            if idx < 0 or idx >= len(self.entries):
                continue
            value = float(score)
            if value < float(min_score):
                continue
            entry = self.entries[idx]
            matches.append(
                {
                    "question": entry["question"],
                    "answer": entry["answer"],
                    "metadata": entry.get("metadata", {}),
                    "score": value,
                }
            )
        return matches

    def get_answer(self, user_input, min_score=0.45):
        match = self.get_match(user_input)
        if not match:
            return None
        if float(match["score"]) < float(min_score):
            return None
        return match