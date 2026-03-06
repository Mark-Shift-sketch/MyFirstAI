from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Brain:
    def __init__(self, dataset):
        cleaned_pairs = []
        for question, answer in dataset:
            q = question.strip().lower()
            a = answer.strip()
            if q and a:
                cleaned_pairs.append((q, a))

        self.questions = [q for q, _ in cleaned_pairs]
        self.answers = [a for _, a in cleaned_pairs]

        if not self.questions:
            self.questions = ["hello"]
            self.answers = ["Hello. How can I assist you?"]

        # Include simple bigrams to improve short assistant-style intent matching.
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.q_vectors = self.vectorizer.fit_transform(self.questions)

    def get_answer(self, user_input, min_score=0.25):
        prompt = user_input.strip().lower()
        if not prompt:
            return None

        user_vector = self.vectorizer.transform([prompt])
        similarity = cosine_similarity(user_vector, self.q_vectors)
        index = int(similarity.argmax())
        score = float(similarity[0][index])

        if score < float(min_score):
            return None
        return self.answers[index]