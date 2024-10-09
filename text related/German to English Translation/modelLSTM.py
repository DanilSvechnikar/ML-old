import torch.nn as nn
import torch
import random


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size,
                 num_layers, dropout_num):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_num)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_num)

    def forward(self, x):
        # x.shape: (seq_len, N)

        embedding = self.dropout(self.embedding(x))
        # embedding.shape: (seq_len, N, emb_size)

        _, (hidden, cell) = self.rnn(embedding)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size,
                 output_size, num_layers, dropout_num):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_num)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_num)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state, cell_state):
        # x.shape: (N), but we want (1, N)
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embedding.shape: (1, N, emb_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden_state, cell_state))
        # outputs.shape: (1, N, hidden_size)

        predictions = self.fc(outputs)
        # predictions.shape: (1, N, vocab_len)

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, target_vocab_size):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.target_vocab_size = target_vocab_size

        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.num_layers == decoder.num_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]

        outputs = torch.zeros(target_len, batch_size, self.target_vocab_size).to(self.device)

        hidden, cell = self.encoder(source)

        # Grab start token
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output
            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
