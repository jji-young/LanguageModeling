import dataset
from dataset import Shakespeare
import torch
import torch.nn as nn
import model
from model import CharRNN, CharLSTM
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import time
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import os
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datasets = Shakespeare(input_file='shakespeare_train.txt')
chars = datasets.chars

def train(model, trn_loader, device, criterion, optimizer, model_name):
    model.to(device)
    model.train()

    trn_loss = 0

    for input, target in trn_loader:
        input, target = input.to(device), target.to(device)

        batch_size = input.size(0)
        
        if model_name == 'rnn':
            hidden = model.init_hidden(batch_size=batch_size).to(device)
        elif model_name == 'lstm':
            hidden = model.init_hidden(batch_size=batch_size)
            hidden = (hidden[0].to(device), hidden[1].to(device))
        
        optimizer.zero_grad()
        output, hidden = model(input, hidden)

        loss = criterion(output.view(-1, model.vocab_size), target.view(-1))
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()

    trn_loss = trn_loss / len(trn_loader)

    return trn_loss

def validate(model, val_loader, device, criterion, model_name):
    model.to(device)
    model.eval()

    val_loss = 0

    with torch.no_grad():
        for input, target in val_loader:
            input, target = input.to(device), target.to(device)

            batch_size = input.size(0)
            
            if model_name == 'rnn':
                hidden = model.init_hidden(batch_size=batch_size).to(device)
            elif model_name == 'lstm':
                hidden = model.init_hidden(batch_size=batch_size)
                hidden = (hidden[0].to(device), hidden[1].to(device))

            output, hidden = model(input, hidden)
            loss = criterion(output.view(-1, model.vocab_size), target.view(-1))

            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)

    return val_loss

def main(epochs, model_name, batch_size, hidden_dim):
    data = datasets

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plot_save_dir = './img'
    model_save_dir = './model'

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    vocab_size = len(data.chars)

    index_list = list(range(len(data)))
    np.random.shuffle(index_list)
    split = int(np.floor(0.8 * len(data)))

    train_sampler = SubsetRandomSampler(indices=index_list[:split])
    valid_sampler = SubsetRandomSampler(indices=index_list[split:])

    train_dataloader = DataLoader(dataset=data, batch_size=batch_size, sampler=train_sampler)
    valid_dataloader = DataLoader(dataset=data, batch_size=batch_size, sampler=valid_sampler)

    train_losses = []
    val_losses = []

    # RNN
    if model_name == 'rnn':
        print(f'Training RNN using {device}...')
        
        model = CharRNN(vocab_size=vocab_size, hidden_dim=hidden_dim)
        criterion  = CrossEntropyLoss()
        optimizer = Adam(params=model.parameters())
        
        for epoch in range(epochs):
            print(f'Epoch: [{epoch+1}/{epochs}]')

            train_loss = train(model=model, trn_loader=train_dataloader, device=device, criterion=criterion, optimizer=optimizer, model_name=model_name)
            train_losses.append(train_loss)

            val_loss = validate(model=model, val_loader=valid_dataloader, device=device, criterion=criterion, model_name=model_name)
            val_losses.append(val_loss)

            print(f'Train Loss: {train_loss}', '\t', f'Valid Loss: {val_loss}')
    
    # LSTM
    if model_name == 'lstm':
        print(f'Training LSTM using {device}...')
        
        model = CharLSTM(vocab_size=vocab_size, hidden_dim=hidden_dim)
        criterion = CrossEntropyLoss()
        optimizer = Adam(params=model.parameters())

        for epoch in range(epochs):
            print(f'Epoch: [{epoch+1}/{epochs}]')

            train_loss = train(model=model, trn_loader=train_dataloader, device=device, criterion=criterion, optimizer=optimizer, model_name=model_name)
            train_losses.append(train_loss)

            val_loss = validate(model=model, val_loader=valid_dataloader, device=device, criterion=criterion, model_name=model_name)
            val_losses.append(val_loss)

            print(f'Train Loss: {train_loss}', '\t', f'Valid Loss: {val_loss}')

    # Plot
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')

    if model_name == 'rnn':
        plt.title('[CharRNN] Cross Entropy Loss by Epochs')
    elif model_name == 'lstm':
        plt.title('[CharLSTM] Cross Entropy Loss by Epochs')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)
    plt.savefig(os.path.join(plot_save_dir, f'{model_name}.png'))

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    torch.save(model.state_dict(), os.path.join(model_save_dir, f'{model_name}.pth'))

    return device, model, data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--model_name', type=str, required=True, help='rnn or lstm')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)  # Set hidden_dim to 128
    args = parser.parse_args()

    epochs = args.epochs
    model_name = args.model_name
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim

    device, model, dataset = main(epochs=epochs, model_name=model_name, batch_size=batch_size, hidden_dim=hidden_dim)
