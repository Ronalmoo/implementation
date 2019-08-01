import torch
import torch.nn as nn
from model.ops import Permute, FullyConnected, ConvBlock
from gluonnlp import Vocab


class VDCNN(nn.Module):
    """Very Deep CNN"""
    def __init__(self, num_classes: int, embedding_dim: int, k_max: int, vocab: Vocab) -> None:
        """

        :param num_classes: the number of classes
        :param embedding_dim: Dimension of embedding vector for token
        :param k_max: k_max pooling
        :param vocab: gluonnlp
        """
        super(VDCNN, self).__init__()
        self._structure = nn.Sequential(nn.Embedding(16, embedding_dim, 0),
                                        Permute(),
                                        nn.Conv1d(embedding_dim, 64, kernel_size=(3, 1, 1)),
                                        ConvBlock(64, 64),
                                        ConvBlock(64, 64),
                                        nn.MaxPool1d(2, 2),
                                        ConvBlock(64, 128),
                                        ConvBlock(128, 128),
                                        nn.MaxPool1d(2, 2),
                                        ConvBlock(128 , 256),
                                        ConvBlock(256, 256),
                                        nn.MaxPool1d(2, 2),
                                        ConvBlock(256, 512),
                                        ConvBlock(512, 512),
                                        nn.AdaptiveMaxPool1d(k_max),
                                        FullyConnected())

        self._classifier = nn.Sequential(nn.Linear(512 * k_max, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048, num_classes))

    def forward(self, x):
        feature = self._structure(x)
        score = self._classifier(feature)
        return score
