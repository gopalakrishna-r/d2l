import matplotlib.pyplot as plt
import tensorflow as tf
from d2l import tensorflow as d2l

from RNN.text_preprocessing.DataReader import read_time_machine
from RNN.text_preprocessing.Tokenizer import tokenize
from RNN.text_preprocessing.Vocabulary import Vocab


def load_corpus_time_machine(min_frequencies, max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens, min_freq=min_frequencies)

    time_machine_corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        time_machine_corpus = time_machine_corpus[:max_tokens]
    return len(vocab)


frequency_range = tf.range(0, 1000, dtype=tf.float32)
vocab_size = [load_corpus_time_machine(min_frequencies=min_freq) for min_freq in frequency_range]
d2l.plot(frequency_range, vocab_size, 'minimum frequencies', 'vocab size', xlim=[0, 1000], figsize=(6, 3))
plt.show()
