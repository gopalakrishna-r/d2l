from d2l import tensorflow as d2l

from MRNN.machine_translation.MTFraEng import MTFraEng


@d2l.add_to_class(MTFraEng)
def _preprocess(self,text):
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ''

    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]
    return ''.join(out)
