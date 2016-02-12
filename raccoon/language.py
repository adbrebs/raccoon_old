"""
Extensions and quantities designed for NLP
"""

import os

import numpy as np
import scipy.spatial.distance as spd
from scipy.stats import spearmanr
import heapq

from utils import build_embedding_dictionary
from extensions import Extension
from quantities import MonitoredQuantity


class MostSimilarWords(Extension):
    """
    Computes the nearest neighbours of some words in an embedding matrix
    """
    def __init__(self, freq, words, embedding_matrix, knn, vocab, inv_vocab):
        Extension.__init__(self, 'Nearest neighbours of words', freq)
        self.words = words
        self.embedding_matrix = embedding_matrix
        self.knn = knn
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.word_ids = []
        for word in words:
            self.word_ids.append(self.vocab[word])

    def find_knn(self, word_id, npy_emb_matrix):
        distances = spd.cdist(npy_emb_matrix[word_id, None],
                              npy_emb_matrix, 'cosine')[0]
        idxs = heapq.nsmallest(self.knn, range(len(distances)), distances.take)
        return [self.inv_vocab[idx] for idx in idxs]

    def execute_virtual(self, batch_id):
        npy_emb_matrix = self.embedding_matrix.get_value()
        strs = []
        for word, word_id in zip(self.words, self.word_ids):
            neighbours = self.find_knn(word_id, npy_emb_matrix)
            strs.append("{}: {}".format(word, neighbours))
        return strs


class WordEmbeddingScores(MonitoredQuantity):
    """
    Parts are taken from the MILA servers. I don't know the precise author.
    """
    def __init__(self, embedding_matrix, inv_vocab):
        name_or_names = ['simlex', 'conc', 'simlex333', 'USF', 'wordsim', 'MEN']
        MonitoredQuantity.__init__(self, name_or_names)
        self.embedding_matrix = embedding_matrix
        self.inv_vocab = inv_vocab

        self.embedding_matrix = embedding_matrix
        self.inv_vocab = inv_vocab

        path_eval = os.environ.get(
            'WORD_EMB_EVAL', '/data/lisatmp3/devincol/embeddings/Evaluations')

        with open(os.path.join(path_eval, "SimLex999_FINAL.txt")) as f:
            lines = [x.split() for x in f.read().splitlines()]

        self.gold_data_sim = {(x[0].lower(), x[1].lower()): float(x[-2])
                              for x in lines[1:]}
        self.gold_data_conc = {(x[0].lower(), x[1].lower()): float(x[-2])
                               for x in lines[1:] if int(x[3])==4}
        self.gold_data_assoc = {(x[0].lower(), x[1].lower()): float(x[-3])
                                for x in lines[1:]}

        # SimAssoc333
        assoc_pairs = sorted([[p, self.gold_data_assoc[p]]
                              for p in self.gold_data_assoc.keys()],
                             key=lambda x: x[1], reverse=True)
        high_assoc_pairs = [x[0] for x in assoc_pairs[:333]]
        self.gold_data_simassoc = {p: self.gold_data_sim[p]
                                   for p in high_assoc_pairs}


        # equivalent data from WS-353
        with open(os.path.join(path_eval, "combined.tab")) as f:
            lines = [x.split('\t') for x in f.read().splitlines()]

        self.gold_data_ws = {(x[0], x[1]): float(x[-1]) for x in lines[1:]}

        with open(os.path.join(path_eval,
                               "MEN/MEN/MEN_dataset_natural_form_full")) as f:
            lines = [x.split() for x in f.read().splitlines()]

        self.gold_data_MEN = {(x[0], x[1]): float(x[-1]) for x in lines[1:]}

    def evaluate(self, model_data, gold_standard, Vocab):
        """
        evaluate a model when the model data is stored as a word:embedding
        dictionary / and the gold_standard is stored as a (w1,w2):score
        dictionary
        """
        overlap = [p for p in gold_standard.keys() if p[0] in Vocab and p[1]
                   in Vocab]
        model_predictions = [1-spd.cosine(model_data[p[0]], model_data[p[1]])
                             for p in overlap]
        gold_predictions = [gold_standard[(p[0], p[1])] for p in overlap]
        return spearmanr(model_predictions, gold_predictions)[0]

    def calculate(self, *inputs):

        raw_data = build_embedding_dictionary(
            self.embedding_matrix.get_value(), self.inv_vocab)
        vocab = [key for key in raw_data]
        model_data = {}
        word_set = set()
        for item in vocab:
            item_lower = item.decode('utf-8').lower().encode('utf-8')
            if item_lower not in model_data:
                if item in raw_data:
                    model_data[item_lower] = raw_data[item]
                    word_set.add(item)
                elif item_lower in raw_data:
                    model_data[item_lower] = raw_data[item_lower]
                    word_set.add(item_lower)
            else:
                if (item in raw_data) and (item not in word_set) :
                    model_data[item_lower] += raw_data[item]
                    word_set.add(item)
                elif (item_lower in raw_data) and (item_lower not in word_set):
                    model_data[item_lower] += raw_data[item_lower]
                    word_set.add(item_lower)
        vocab = model_data.keys()

        for w, v in model_data.iteritems():
                model_data[w] = v / np.sqrt((v**2.).sum())

        simlex_result = self.evaluate(model_data, self.gold_data_sim, vocab)
        conc_result = self.evaluate(model_data, self.gold_data_conc, vocab)
        simlex_assoc333_result = self.evaluate(model_data,
                                               self.gold_data_simassoc, vocab)
        USF_result = self.evaluate(model_data, self.gold_data_assoc, vocab)
        wordsim_result = self.evaluate(model_data, self.gold_data_ws, vocab)
        MEN_result = self.evaluate(model_data, self.gold_data_MEN, vocab)

        return [simlex_result, conc_result, simlex_assoc333_result,
                USF_result, wordsim_result, MEN_result]
