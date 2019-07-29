import pandas as pd
import gluonnlp as nlp
from mecab import MeCab
import torch
import torchtext
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Corpus(Dataset):
    def __init__(self, filepath, transform_fn, min_length, pad_val=1):
        self._corpus = pd.read_csv(filepath, sep='\t').loc[:, ['document', 'label']]
        self._transform = transform_fn
        self._min_length = min_length
        self._pad_val = pad_val

    def __len__(self):
        return len(self._corpus)

    def __getitem__(self, idx):
        token2indices = self._transform(self._corpus.iloc[idx]['document'])
        if len(token2indices) < self._min_length:
            token2indices = token2indices + (self._min_length - len(token2indices)) * [self._pad_val]
        token2indices = torch.tensor(token2indices)
        label = torch.tensor(self._corpus.iloc[idx]['label'])
        return token2indices, label

def batchify(data):
    """
    custom collate_fn for DataLoader
    :param data: list of torch.Tensors
    :return: data (tuple): tuple of torch.Tensors
    """
    indices, labels = zip(*data)
    indices = pad_sequence(indices, batch_first=True, padding_value=1)
    labels = torch.stack(labels, 0)
    return indices, labels
