import itertools
import pickle
import gluonnlp as nlp
import pandas as pd
from mecab import MeCab
from convolutional_neural_network_for_sentence_classification.model.utils import Vocab
from convolutional_neural_network_for_sentence_classification.utils import Config
from collections import Counter


# loading data set
data_config = Config('data/config.json')
tr = pd.read_csv(data_config.train, sep='\t').loc[:, ['document', 'label']]

# extracting morph in sentences
split_fn = MeCab().morphs
list_of_tokens =tr['document'].apply(split_fn).tolist()

# generating the vocab
min_freq = 10
token_counter = Counter(itertools.chain.from_iterable(list_of_tokens))
list_of_tokens = [token_counter[0] for token_counter in token_counter.items() if token_counter[1] >= min_freq]
list_of_tokens = sorted(list_of_tokens)
list_of_tokens.insert(0, '<pad>')
list_of_tokens.insert(0, '<unk>')

tmp_vocab = nlp.Vocab(counter=Counter(list_of_tokens), min_freq=1, bos_token=None, eos_token=None)

# connecting SISG embedding with vocab
ptr_embedding = nlp.embedding.create('fasttext', source='wiki.ko')
tmp_vocab.set_embedding(ptr_embedding)
array = tmp_vocab.embedding.idx_to_vec.asnumpy()

vocab = Vocab(list_of_tokens, padding_token='<pad>', unknown_token='<unk>', bos_token=None, eos_token=None)
vocab.embedding = array

# saving vocab
with open('data/vocab.pkl', mode='wb') as io:
    pickle.dump(vocab, io)
data_config.vocab = 'data/vocab.pkl'
data_config.save('data/config.json')
