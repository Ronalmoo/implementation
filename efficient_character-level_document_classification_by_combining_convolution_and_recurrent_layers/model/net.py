import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ops import Embedding, Conv1d, MaxPool1d, Linker, BiLSTM
from gluonnlp import Vocab


class ConvRec(nn.Module):
    def __init__(self, num_classes, embedding_dim, hidden_dim, vocab=Vocab):
        """
        Intantiating ConvRec class

        :param num_classes: the number of classes
        :param embedding_dim: the dimension of embedding vector for token
        :param hidden_dim: the dimension of hidden_layer
        :param vocab: gluonnlp.Vocab
        """
        super(ConvRec, self).__init__()
        self._ops = nn.Sequential(
            Embedding(len(vocab), embedding_dim, vocab.to_indices(vocab.padding_token),
                      permuting=True, tracking=True),
            Conv1d(embedding_dim, hidden_dim, 5, 1, 1, F.relu, tracking=True),
            MaxPool1d(2, 2, tracking=True),
            Conv1d(hidden_dim, hidden_dim, 3, 1, 1, F.relu, tracking=True),
            MaxPool1d(2, 2, tracking=True),
            Linker(permuting=True),
            BiLSTM(hidden_dim, hidden_dim, using_sequence=False),
            nn.Dropout(),
            nn.Linear(in_features=2 * hidden_dim, out_features=num_classes)
        )
        self.apply(self._init_weights)

    def forward(self, x):
        score = self._ops(x)
        return score

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
