import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers=1):
        super(CharRNN, self).__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.RNN(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, input, hidden):
        embedded = self.embed(input)
        output, hidden = self.rnn(embedded, hidden)
        output = output.contiguous().view(-1, self.hidden_dim)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        initial_hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return initial_hidden

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers=1):
        super(CharLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, hidden):
        embedded = self.embed(input)
        output, hidden = self.lstm(embedded, hidden)
        output = output.contiguous().view(-1, self.hidden_dim)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        initial_hidden = (
            torch.zeros(self.n_layers, batch_size, self.hidden_dim),
            torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        )
        return initial_hidden
