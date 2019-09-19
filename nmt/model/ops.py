import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy

import random
import math
import time


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, dec_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # |src| = [src sent len, batch_size]
        embedded = self.dropout(self.embedding(src))

        # embedded = [src_sent_len, batch_size, emb_dim]
        outputs, hidden = self.rnn(embedded)

        # outputs = [src_sent_len, batch_size, hid_dim * num_directions]
        # hidden = [n_layers * num_directions, batch_size, hid_dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden[-2, :, : ] is the last of the forwards RNN
        # hidden[-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        # encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [src_sent_len, batch_size, enc_hid_dim * 2]
        # hidden = [batch_size, dec_hid_dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Paramer(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch_size, dec_hid_dim]
        # encoder_outputs = [src_sent_len, batch_size, enc_hid_dim * 2]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repear(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch_size, src_sent_len, dec_hid_dim]
        # encoder_outputs = [batch_size, src_sent_len, enc_hid_dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch_size, src_sent_len, dec_hid_dim]
        energy = energy.permute(0, 2, 1)
        # energy = [batch_size, dec_hid_dim, src_sent_len]
        # v = [dec_hid_dim]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        # v = [batch_size, 1, dec_hid_dim]
        attention = torch.bmm(v, energy).squeeze(1)
        # attention = [batch_size, src_len]
        return F.softmax(attention, dim=1)



