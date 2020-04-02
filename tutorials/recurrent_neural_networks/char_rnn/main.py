import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tqdm import tqdm

with open('data/anna.txt', 'r') as f:
    text = f.read()

# Tokenization
chars = tuple(set(text))
idx2char = dict(enumerate(chars))
char2idx = {char: idx for idx, char in idx2char.items()}
encoded = np.array([char2idx[char] for char in text])


def one_hot_encode(arr, vocab_size):
    return keras.utils.to_categorical(arr, vocab_size)


def get_batches(arr, batch_size, seq_length):
    """Splits arr into multiple sequences, given by batch_size.
    Each of our sequences will be seq_length long."""

    chars_per_batch = batch_size * seq_length
    num_batches = len(arr) // chars_per_batch

    # Keep only enough characters to make full batches
    arr = arr[:num_batches * chars_per_batch]
    # Reshape into batch_size rows
    arr = arr.reshape(batch_size, -1)

    for n in range(0, arr.shape[1], seq_length):
        x = arr[:, n: n + seq_length]
        y = np.zeros_like(x)

        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]

        yield x, y


class CharRNN(nn.Module):

    def __init__(self, tokens, hidden_size=256, num_layers=2, drop_prob=0.1, batch_first=True):
        super(CharRNN, self).__init__()
        self.vocab_size = len(tokens)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop_prob = drop_prob

        # Tokenization
        self.chars = tokens
        self.idx2char = dict(enumerate(self.chars))
        self.char2idx = {char: idx for idx, char in self.idx2char.items()}

        # self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        self.lstm = nn.LSTM(self.vocab_size, hidden_size, num_layers,
                            batch_first=batch_first, dropout=drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)

        # Stack up LSTM outputs using view
        # you may need to use contiguous to reshape the output
        x = x.contiguous().view(-1, self.hidden_size)
        x = self.fc(x)
        return x, hidden

    def init_hidden(self, batch_size, device):
        """Initializes hidden state."""
        # Create two new tensors with sizes num_layers x batch_size x hidden_size,
        # initialized to zero, for hidden state and cell state of LSTM
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))


def train(net, data, device, epochs=10, batch_size=10, seq_length=50, lr=1e-3, clip=5, split_ratio=0.1):
    optimizer = optim.Adam(net.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    vocab_size = net.input_size
    net.to(device)

    # Splits data into train, val
    # train_data, test_data = train_test_split(data, test_size=0.3, shuffle=True)

    for epoch in range(epochs):
        hidden = net.init_hidden(batch_size, device)

        # Iterate over data.
        for i, (x, y) in enumerate(get_batches(data, batch_size, seq_length)):
            x = one_hot_encode(x, vocab_size)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = tuple([each.detach() for each in hidden])
            outputs, hidden = net(inputs, hidden)

            loss = criterion(outputs, targets.flatten())
            loss.backward()

            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

        print(f'Epoch: {epoch} loss:{loss}')


def predict(net, char, hidden, device, top_k=None):
    """Returns next characters and hidden states given a character."""

    x = torch.tensor([[net.char2idx[char]]])
    x = x.to(device)

    # detach hidden state from history
    hidden = tuple([each.detach() for each in hidden])
    preds, hidden = net(x, hidden)
    prob = F.softmax(preds, dim=1).cpu()

    if top_k is None:
        top_char = np.arange(net.input_size)
    else:
        prob, top_char = prob.topk(top_k)
        top_char = top_char.numpy().squeeze()

    # select the likely next character with some element of randomness
    prob = prob.numpy().squeeze()
    char = np.random.choice(top_char, p=prob / prob.sum())

    return net.idx2char[char]


if __name__ == '__main__':
    batch_size = 128
    epochs = 20
    seq_length = 100
    hidden_size = 128
    num_layers = 2

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = CharRNN(chars, hidden_size, num_layers)
    print(net)

    train(net, encoded, device, epochs, batch_size, seq_length)
