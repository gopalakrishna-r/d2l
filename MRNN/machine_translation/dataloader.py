import tensorflow as tf
from d2l import tensorflow as d2l

from MRNN.machine_translation.MTFraEng import MTFraEng

d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')


@d2l.add_to_class(MTFraEng)
def _download(self):
    d2l.extract(d2l.download(
        d2l.DATA_URL + 'fra-eng.zip', self.root,
        '94646ad1522d915e7b0f9296181140edcf86a4f5'))
    with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
        return f.read()


@d2l.add_to_class(MTFraEng)
def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
    def _build_array(sentences, vocab, is_tgt=False):
        pad_or_trim = \
            lambda seq, t: (seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
        sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
        if is_tgt:
            sentences = [['<bos>'] + s for s in sentences]
        if vocab is None:
            vocab = d2l.Vocab(sentences, min_freq=2)
        array = tf.constant([vocab[s] for s in sentences])
        valid_len = tf.reduce_sum(tf.cast(array != vocab['<pad>'], tf.int32), 1)
        return array, vocab, valid_len

    src, tgt = self._tokenize(self._preprocess(raw_text), self.num_train + self.num_val)

    src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
    tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
    return (src_array, tgt_array[:, :-1], src_valid_len, tgt_array[:, 1:]), src_vocab, tgt_vocab


@d2l.add_to_class(MTFraEng)
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader(self.arrays, train, idx)
