import torch
import torch.nn as nn
from convolutional_neural_networks_for_sentence_classification_torch.model.ops import MultiChannelEmbedding, ConvolutionLayer, MaxOverTimePooling
from convolutional_neural_networks_for_sentence_classification_torch.model.utils import Vocab


class SenCNN(nn.Module):
    """SenCNN class"""
    def __init__(self, num_classes: int, vocab: Vocab) -> None:
        """
        Instantiating SenCNN class

        :param num_classes: the numer of classes
        :param vocab: the instance of model.utils.Vocab
        """
        super(SenCNN, self).__init__()
        self._embedding = MultiChannelEmbedding(vocab)
        self._convolution = ConvolutionLayer(300, 300)
        self._pooling = MaxOverTimePooling()
        self._dropout = nn.Dropout()
        self._fc = nn.Linear(300, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self._embedding(x)
        fmap = self._convolution(fmap)
        feature = self._pooling(fmap)
        feature = self._dropout(feature)
        score = self._fc(feature)

        return score
    