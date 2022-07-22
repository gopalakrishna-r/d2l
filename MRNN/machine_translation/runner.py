import tensorflow as tf
from dataloader import *
from preprocessor import *
from tokenizer import *
from MRNN.machine_translation.MTFraEng import MTFraEng

data = MTFraEng(batch_size=3)
src, tgt, src_valid_len, label = next(iter(data.train_dataloader()))

print('source', tf.cast(src, tf.int32))
print('decoder input', tf.cast(tgt, tf.int32))
print('source len excluding pad:', tf.cast(src_valid_len, tf.int32))
print('label', tf.cast(label, tf.int32))


@d2l.add_to_class(MTFraEng)
def build(self, src_sentences, tgt_sentences):
    raw_text = '\n'.join([source + '\t' + target for source, target in zip(src_sentences, tgt_sentences)])
    arrays, _, _ = self._build_arrays(raw_text, self.src_vocab, self.tgt_vocab)
    return arrays


src, tgt, _, _ = data.build(['hi .'], ['salut .'])
print('source', data.src_vocab.to_tokens(tf.cast(src[0], tf.int32)))
print('target', data.tgt_vocab.to_tokens(tf.cast(tgt[0], tf.int32)))
