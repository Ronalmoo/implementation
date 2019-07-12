import torch
import torch.nn as nn
import torch.nn.functional as F
from convolutional_neural_networks_for_sentence_classification_torch.model.utils import Vocab
from typing import Tuple


class MultiChannelEmbedding(nn.Module):
    def __init__(self, vocab: Vocab) -> None:
        super(MultiChannelEmbedding, self).__init__()
        self._static = nn.Embedding.from_pretrained(torch.from_numpy(vocab.embedding),
                                                    freeze=True, padding_idx=vocab.to_indices(vocab.padding_token))
        self._non_static = nn.Embedding.from_pretrained(torch.from_numpy(vocab.embedding),
                                                        freeze=True, padding_idx=vocab.to_indices(vocab.padding_token))

    def forward(self, x: torch.Tensor) -> None:
        static = self._static(x).permute(0, 2, 1)
        non_static = self._non_static(x).permute(0, 2, 1)

        return static, non_static


class ConvolutionLayer(nn.Module):
    """ConvolutionLayer class"""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Instantiating Convolution Layer class

        :param in_channels: the number of channels from input feature map
        :param out_channels: the number of channels from output feature map
        """
        super(ConvolutionLayer, self).__init__()
        self._tri_gram = nn.Conv1d(in_channels=in_channels, out_channels=out_channels // 3, kernel_size=3)
        self._tetra_gram = nn.Conv1d(in_channels=in_channels, out_channels=out_channels // 3, kernel_size=4)
        self._penta_gram = nn.Conv1d(in_channels=in_channels, out_channels=out_channels // 3, kernel_size=5)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        static, non_static = x
        tri_fmap = F.relu(self._tri_gram(static)) + F.relu(self._tri_gram(non_static))
        tetra_fmap = F.relu(self._tetra_gram(static)) + F.relu(self._tetra_gram(non_static))
        penta_fmap = F.relu(self._penta_gram(static)) + F.relu(self._penta_gram(non_static))

        return tri_fmap, tetra_fmap, penta_fmap


class MaxOverTimePooling(nn.Module):
    """MaxOverTimePooling class"""
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        tri_fmap, tetra_fmap, penta_fmap = x
        fmap = torch.cat([tri_fmap.max(dim=-1)[0], tetra_fmap.max(dim=-1)[0], penta_fmap.max(dim=-1)[0]], dim=-1)
        return fmap

