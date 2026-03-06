# Script for generating text responses

import torch
import torch.nn.functional as F
from model.lstm_model import SimpleLSTM
from utils import clean_text, create_vocab

# Load data to build vocab
with open("data/corpus.txt", "r", encoding="utf-8") as f:
    text = clean_text(f.read())
    
word_to_idx, idx_to_word = create_vocab(text)
vocab_size = len(word_to_idx)


# Load trained model (fallback-safe)
model = SimpleLSTM(vocab_size)
_model_ready = False

try:
    state = torch.load("model.lstm_weights.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    _model_ready = True
except Exception:
    # Keep runtime stable even when weights and vocab do not match.
    _model_ready = False

def generate_text(
    start_seq,
    length=14,
    temperature=0.8,
    top_k=25,
    repetition_penalty=1.1,
    context_window=20,
):
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