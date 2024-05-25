import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from main import device, datasets
from model import CharRNN, CharLSTM

def generate(model, dataset, device, seed_characters, temperature, num_chars):
    model.eval()
    hidden = model.init_hidden(1)
    samples = seed_characters
    
    x = torch.tensor([dataset.char_to_index[char] for char in seed_characters], dtype=torch.long).to(device)
    
    for _ in range(num_chars):
        x = x.view(1, -1)
        output, hidden = model(x, hidden)
        output = output[-1]
        pred = F.softmax(output / temperature, dim=-1).detach().cpu().numpy()
        next_char_idx = np.random.choice(dataset.vocab_size, p=pred)
        next_char = dataset.index_to_char[next_char_idx]
        samples += next_char
        x = torch.tensor([next_char_idx], dtype=torch.long).to(device)
    
    return samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=str, required=True)
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--num_chars', type=int, default=100)
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()

    seed_characters = args.seed
    temperature = args.temp
    num_chars = args.num_chars
    model_name = args.model_name

    hidden_dim = 128  # Ensure this matches the hidden_dim used in main.py

    if model_name == 'rnn':
        model = CharRNN(vocab_size=len(datasets.chars), hidden_dim=hidden_dim)
    elif model_name == 'lstm':
        model = CharLSTM(vocab_size=len(datasets.chars), hidden_dim=hidden_dim)
    
    model.to(device)
    
    model_save_dir = './model'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    model_path = os.path.join(model_save_dir, f'{model_name}.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path))
    
    generated_text = generate(model=model, dataset=datasets, device=device, seed_characters=seed_characters, temperature=temperature, num_chars=num_chars)
    print(generated_text)
