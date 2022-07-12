from d2l import tensorflow as d2l
import tensorflow as tf
from MRNN.machine_translation.dataloader import read_data_nmt, load_data_nmt
from MRNN.machine_translation.preprocessor import preprocess_nmt, truncate_pad
from MRNN.machine_translation.tokenizer import tokenize_nmt

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2 , num_steps=8)
for X , X_valid_len, Y, Y_valid_len in train_iter:
    print('X', tf.cast(X, tf.int32))
    print('valid lengths for X:', X_valid_len)
    print('Y', tf.cast(Y, tf.int32))
    print('valid lengths for Y:', Y_valid_len)
    break

