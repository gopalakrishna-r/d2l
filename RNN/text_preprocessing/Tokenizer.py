from spacy.lang.en.tokenizer_exceptions import word

from RNN.text_preprocessing.DataReader import read_time_machine


def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('Unexpected token type ' + token)