"""
Script to generate text using the trained LSTM model.
"""
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from model.lstm_model import SimpleLSTM
from core.utils import clean_text, create_vocab


def generate_text(model, seed_text, word_to_idx, idx_to_word, seq_length=5, gen_length=15, temp=0.8):
    model.eval()
    
    # Process seed text
    cleaned_seed = clean_text(seed_text)
    words = cleaned_seed.split()
    
    # We only take the last `seq_length` words or pad if too short
    current_words = words[-seq_length:]
    
    # Map words to indices, using 0 for unknown words
    current_idx = [word_to_idx.get(w, 0) for w in current_words]
    
    # Simple zero-padding if the seed is shorter than seq_length
    while len(current_idx) < seq_length:
        current_idx.insert(0, 0)
        
    generated_output = list(words)
    
    with torch.no_grad():
        for _ in range(gen_length):
            # Shape: (batch=1, seq_len)
            x = torch.tensor([current_idx], dtype=torch.long)
            
            # Predict
            out, hidden = model(x)
            
            # Use only the prediction from the last time step
            logits = out[:, -1, :]  # Shape: (1, vocab_size)
            
            # Apply temperature scaling to control randomness vs. confidence
            logits = logits / temp
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the probability distribution
            next_idx = torch.multinomial(probs, num_samples=1).item()
            
            # Convert back to word
            next_word = idx_to_word.get(next_idx, "<UNK>")
            generated_output.append(next_word)
            
            # Shift the context window forward
            current_idx.append(next_idx)
            current_idx = current_idx[1:]
            
    return " ".join(generated_output)


def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "corpus.txt"
    model_path = base_dir / "models" / "script_lstm_weights.pth"
    
    if not data_path.exists():
        print(f"Data file missing: {data_path}")
        return
    if not model_path.exists():
        print(f"Model weights missing: {model_path}")
        return
        
    # Load corpus to rebuild the exact same vocabulary
    text = clean_text(data_path.read_text(encoding="utf-8"))
    word_to_idx, idx_to_word = create_vocab(text)
    vocab_size = len(word_to_idx)
    print(f"Loaded vocabulary size: {vocab_size}")
    
    # Initialize and load model
    model = SimpleLSTM(vocab_size)
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print("Loaded trained model weights seamlessly.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # Let's test a few sample prompts to see what it learned
    test_prompts = [
        "what is",
        "who is the philippines",
        "the term peguan was"
    ]
    
    print("\n--- Generating Text ---")
    for prompt in test_prompts:
        # High temperature for more variety, lower temperature for safer, rigid predictions
        text_out = generate_text(model, prompt, word_to_idx, idx_to_word, seq_length=5, gen_length=15, temp=0.7)
        print(f"Prompt: '{prompt}'")
        print(f"Result: {text_out}\n")


if __name__ == "__main__":
    main()
