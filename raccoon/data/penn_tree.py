import cPickle
import os

from utils_text import load_data


data_path = os.path.join(os.getenv('DATA_PATH'), 'penntree')
ptb_train_path = os.path.join(data_path, 'ptb.train.txt.gz')
ptb_valid_path = os.path.join(data_path, 'ptb.valid.txt.gz')
ptb_test_path = os.path.join(data_path, 'ptb.test.txt.gz')


if __name__ == '__main__':

    vocab_map = {}
    vocab_idx = [0]

    train_data, freq = load_data(ptb_train_path, vocab_map, vocab_idx, 10000)
    valid_data, _ = load_data(ptb_valid_path, vocab_map, vocab_idx, 10000)
    test_data, _ = load_data(ptb_test_path, vocab_map, vocab_idx, 10000)

    def dump(object, filename):
        cPickle.dump(object,
                     open(os.path.join(data_path, filename), 'wb'))

    dump(train_data, 'train.pkl')
    dump(valid_data, 'valid.pkl')
    dump(test_data, 'test.pkl')
    dump(freq, 'freq.pkl')
    dump(vocab_map, 'vocab_map.pkl')
