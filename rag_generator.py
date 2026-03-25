from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class LocalInstructionGenerator:
    def __init__(self, model_name="google/flan-t5-small"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.ready = False
        self._load_model()

    def _load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.ready = True
        except Exception:
            self.ready = False

    def generate(self, question, facts, intent="general", max_new_tokens=96):
        if not self.ready:
            return ""

        context = "\n".join(f"- {fact}" for fact in facts[:5])
        prompt = (
            "You are JARVIS, a grounded assistant. "
            "Use only the provided facts to answer. "
            "If facts are insufficient, answer with 'I am not fully certain based on current grounded data.'\n"
            f"Intent: {intent}\n"
            f"Question: {question}\n"
            f"Grounding facts:\n{context if context else '- none'}\n"
            "Answer briefly and clearly:"
        )

        tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        output_ids = self.model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=2,
            repetition_penalty=1.1,
        )
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return str(answer).strip()
