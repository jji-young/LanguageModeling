import numpy as np
import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    def __init__(self, input_file, chunk_size=30, sequence_length=30):
        super().__init__()
        
        with open(input_file, 'r', encoding='utf-8') as file:
            self.data = file.read()

        self.chunk_size = chunk_size
        self.sequence_length = sequence_length
        self.chars = sorted(list(set(self.data)))
        self.vocab_size = len(self.chars)

        self.char_to_index = {char: i for i, char in enumerate(self.chars)}
        self.index_to_char = {i: char for i, char in enumerate(self.chars)}
        self.indices = [self.char_to_index[ch] for ch in self.data]
        self.chunks = []

        for i in range(0, len(self.indices) - self.sequence_length, 1):
            self.chunks.append((self.indices[i:i+self.sequence_length], self.indices[i+1:i+self.sequence_length+1]))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_array, target_array = self.chunks[idx]

        input_tensor = torch.tensor(input_array, dtype=torch.long)
        target_tensor = torch.tensor(target_array, dtype=torch.long)

        return input_tensor, target_tensor
    
    def onehot_encoding(self, sequence):
        onehot = np.zeros((len(sequence), self.vocab_size), dtype=np.float32)
        
        for i, idx in enumerate(sequence):
            onehot[i, idx] = 1.0
        return onehot

if __name__ == '__main__':
    data = Shakespeare('shakespeare_train.txt', chunk_size=30, sequence_length=30)
