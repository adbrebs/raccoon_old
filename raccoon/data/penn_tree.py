import os
from itertools import count
import cPickle

import numpy
from numpy.lib.stride_tricks import as_strided


path = '/Users/adeb/data/PennTreebankCorpus'


def get_data():
    return cPickle.load(open('pt_data.pkl'))


def decrypt_data(data, vocab):
    inv_vocab = {j:i for i, j in vocab.iteritems()}
    return ''.join(inv_vocab[i] for i in data)


# def create_n_gram_generator(subset, batch_size, context_size, vocab):
#     eos




# def get_ngram_generator(dataset, batch_size, context_size, times=None):
#     """
#
#     """
#     dataset = numpy.array(dataset)
#     data = as_strided(dataset, shape=(dataset.size - context_size, context_size + 1),
#                       strides=(dataset.itemsize, dataset.itemsize))
#     dataset = IndexableDataset({'features': data[:, :context_size],
#                                 'targets': data[:, context_size:]})
#
#     stream = DataStream(
#         dataset, iteration_scheme=ShuffledExampleScheme(dataset.num_examples))
#     batch_stream = Batch(
#         stream, iteration_scheme=ConstantScheme(batch_size, times=times),
#         strictness=1)
#
#     def reshape(batch):
#         return batch[0].astype("int32"), batch[1].astype("int32")
#
#     return Mapping(batch_stream, reshape)
#
#
# def create_stream(cf):
#     vocab = numpy.load(open(os.path.join(path, "dictionaries.npz")))['unique_words']
#
#     train, valid, test, counts, _, index_mapping = get_data(None, cf.vocab_size)
#
#     train_batch_stream = get_ngram_generator(train, cf)
#     valid_batch_stream = get_ngram_generator(valid, cf)
#     # valid_rare_stream = get_ngram_generator(valid, cf, times=10)
#     valid_rare_stream = get_ngram_generator(valid, cf)
#     test_batch_stream = get_ngram_generator(test, cf)
#
#     vocab = {vocab[i]:j for i, j in index_mapping.iteritems()}
#
#     return train_batch_stream, valid_batch_stream, test_batch_stream, \
#            valid_rare_stream, vocab, counts