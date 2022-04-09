from RNN.text_preprocessing.DataReader import read_time_machine
from RNN.text_preprocessing.Tokenizer import tokenize
from RNN.text_preprocessing.Vocabulary import Vocab


def load_corpus_time_machine(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)

    time_machine_corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        time_machine_corpus = time_machine_corpus[:max_tokens]
    return time_machine_corpus, vocab


corpus, vocab = load_corpus_time_machine()
print(f'length of corpus:{len(corpus)} vocab:{len(vocab)}')
