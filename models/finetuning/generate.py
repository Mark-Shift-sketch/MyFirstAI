"""Utilities for runtime text generation with a reloadable LSTM backend."""

import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.lstm_model import SimpleLSTM
from core.utils import clean_text, create_vocab


word_to_idx = {}
idx_to_word = {}
model = None
_model_ready = False


def _safe_corpus_text():
    try:
        with open("data/corpus.txt", "r", encoding="utf-8") as file_handle:
            return clean_text(file_handle.read())
    except OSError:
        return ""


def _initialize_generator():
    global word_to_idx, idx_to_word, model, _model_ready

    corpus_text = _safe_corpus_text()
    if not corpus_text.strip():
        corpus_text = "jarvis assistant ready to help"

    word_to_idx, idx_to_word = create_vocab(corpus_text)
    vocab_size = max(1, len(word_to_idx))
    model = SimpleLSTM(vocab_size)
    _model_ready = False

    try:
        state = torch.load("model.lstm_weights.pth", map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _model_ready = True
    except Exception:
        # Keep runtime stable even when weights and vocabulary mismatch.
        _model_ready = False


def reload_generator():
    _initialize_generator()
    return _model_ready

def generate_text(
    start_seq,
    length=14,
    temperature=0.8,
    top_k=25,
    repetition_penalty=1.1,
    context_window=20,
):
    if model is None:
        _initialize_generator()

    if not _model_ready:
        return "I am ready to assist."

    prompt_words = [w for w in start_seq.lower().split() if w.strip()]
    if not prompt_words:
        prompt_words = ["jarvis"]

    context_ids = [word_to_idx.get(w, 0) for w in prompt_words]
    generated_ids = []

    for _ in range(max(1, int(length))):
        input_ids = context_ids[-context_window:]
        x = torch.tensor([input_ids], dtype=torch.long)

        with torch.no_grad():
            out, _ = model(x, None)

        logits = out[0, -1, :].clone()

        if repetition_penalty > 1.0:
            for token_id in set(generated_ids[-10:]):
                logits[token_id] = logits[token_id] / repetition_penalty

        logits = logits / max(temperature, 1e-5)

        if top_k and 0 < top_k < logits.size(0):
            top_values, top_indices = torch.topk(logits, top_k)
            probs = F.softmax(top_values, dim=-1)
            chosen = torch.multinomial(probs, num_samples=1).item()
            next_id = int(top_indices[chosen])
        else:
            probs = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        generated_ids.append(next_id)
        context_ids.append(next_id)

    words = [idx_to_word.get(i, "") for i in generated_ids]
    words = [w for w in words if w and w not in {".", ","}]

    output = " ".join(words).strip()
    if not output:
        return "I can assist with commands and brief answers"

    return output


_initialize_generator()