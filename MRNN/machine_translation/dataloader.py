import os

from d2l import tensorflow as d2l

from MRNN.machine_translation.preprocessor import preprocess_nmt, build_array_nmt
from MRNN.machine_translation.tokenizer import tokenize_nmt

d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')


def read_data_nmt():
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r') as f:
        return f.read()


def load_data_nmt(batch_size, num_steps, num_examples = 600):
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, reserved_tokens=['<pad>', '<eos>', '<bos>'], min_freq=2)
    target_vocab = d2l.Vocab(target, reserved_tokens=['<pad>', '<eos>', '<bos>'], min_freq=2)
    print(f'vocab length for source {len(src_vocab)} and target {len(target_vocab)} for num_examples : {num_examples}')
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    target_array, target_valid_len = build_array_nmt(target, target_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, target_array, target_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, target_vocab


for examples in [200, 400, 600, 800, 1000 ]:
    load_data_nmt(batch_size= 2, num_steps=8, num_examples=examples)