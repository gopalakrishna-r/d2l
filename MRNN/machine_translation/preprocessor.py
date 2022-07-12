import tensorflow as tf
def preprocess_nmt(text):
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ''

    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]

    return ''.join(out)


def truncate_pad(line, num_steps, padding_token):
    return line[:num_steps] if len(line) > num_steps else line + [padding_token] * (num_steps - len(line))


def build_array_nmt(lines, vocab, num_steps):
    lines = [vocab[line] for line in lines]
    lines = [line + [vocab['<eos>']] for line in lines]
    array = tf.constant([truncate_pad(line, num_steps, vocab['<pad>']) for line in lines])
    valid_len = tf.reduce_sum(tf.cast(array != vocab['<eos>'], tf.int32), 1)
    return array, valid_len



