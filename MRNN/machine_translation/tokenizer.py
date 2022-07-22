from d2l import tensorflow as d2l

from MRNN.machine_translation.MTFraEng import MTFraEng


@d2l.add_to_class(MTFraEng)
def _tokenize(self, text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
            target.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
    return source, target


