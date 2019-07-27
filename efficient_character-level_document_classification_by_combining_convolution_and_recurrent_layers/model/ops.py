import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


class Embedding(nn.Module):
    def __init__(self, num_embedding, embedding_dim, padding_idx=1, permuting=True,
                 tracking=True):
        """
        Instantiating Embedding class

        :param num_embedding: the number of vocabulary size
        :param embedding_dim: the dimension of embedding vector
        :param padding_idx: denote padding_idx to "<pad>" token
        :param permuting: permuting (n, l, c) -> (n, c, l). Default: True
        :param tracking: tracking length of sequence. Default: True
        """
        super(Embedding, self).__init__()
        self._tracking = tracking
        self._permuting = permuting
        self._padding_idx = padding_idx
        self._ops = nn.Embedding(num_embedding, embedding_dim, self._padding_idx)

    def forward(self, x):
        fmap = self._ops(x).permute(0, 2, 1) if self._permuting else self._ops(x)

        if self._tracking:
            fmap_length = torch.ne(self._padding_idx).sum(dim=1)
            return fmap, fmap_length
        else:
            return fmap


class MaxPool1d(nn.Module):
    """MaxPool1d class"""

    def __init__(self, kernel_size, stride, tracking=True):
        """
        Instantiating MaxPool1d class

        :param kernel_size: the kernel size of 1d max pooling
        :param stride: the stride of 1d max pooling
        :param tracking: tracking length of sequence. Default: True
        """
        super(MaxPool1d, self).__init__()
        self._kernel_size = kernel_size
        self._stride = stride
        self._tracking = tracking

        self._ops = nn.MaxPool1d(kernel_size, stride)

    def forward(self, x):
        if self._tracking:
            fmap, fmap_length = x
            fmap = self._ops(fmap)
            fmap_length = (fmap_length - (self._kernel_size - 1) - 1) / self._stride + 1
            return fmap, fmap_length
        else:
            fmap = self._ops(x)
            return fmap


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1,
                 activation=F.relu, tracking=True):
        """
        Instantiating Conv1d class

        :param in_channels: the number of channels in the input feature map
        :param out_channels: the number of channels in the output feature map
        :param kernel_size: the size of kernel
        :param stride: stride of the convolution. Default: 1
        :param padding: zero-padding added to both sides of the input. Default: 1
        :param activation: acfivation function. Default: ReLu(F.relu)
        :param tracking: tracking length of sequence. Default: True
        """
        super(Conv1d, self).__init__()
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._ops = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self._activation = activation
        self._tracking = tracking

    def forward(self, x):
        if self._tracking:
            fmap, fmap_length = x
            fmap_length = (fmap_length + 2 * self._padding - (self._kernel_size - 1) - 1) / self._stride + 1
            fmap = self._activation(self._ops(fmap)) if self._activation is not None else self._ops(fmap)
            return fmap, fmap_length
        else:
            fmap = self._activation(self._ops(x)) if self._activation is not None else self._ops(x)
            return fmap


class Linker(nn.Module):
    """Linker class"""

    def __init__(self, permuting=True):
        """
        Instantiating Linker class

        :param permuting: permuting(n, c, l) -> (n, l, c).   Default: True
        """
        super(Linker, self).__init__()
        self._permuting = permuting

    def forward(self, x):
        fmap, fmap_length = x
        fmap = fmap.permute(0, 2, 1) if self._permuting else fmap
        return pack_padded_sequence(fmap, fmap_length, batch_first=True, enforce_sorted=False)


class BiLSTM(nn.Module):
    """BiLSTM class"""

    def __init__(self, input_size, hidden_size, using_sequence=True):
        super(BiLSTM, self).__init__()
        self._using_sequence = using_sequence
        self._ops = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x):
        outputs, hc = self._ops(x)

        if self._using_sequence:
            hiddens = pad_packed_sequence(outputs)[0].permute(1, 0, 2)
            return hiddens
        else:
            feature = torch.cat([*hc[0]], dim=1)
            return feature
