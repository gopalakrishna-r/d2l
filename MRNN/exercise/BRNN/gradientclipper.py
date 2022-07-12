import tensorflow as tf


def gradient_clipping(gradients, theta):
    theta = tf.constant(theta, dtype=tf.float32)
    new_gradients = [tf.convert_to_tensor(grad) if isinstance(grad, tf.IndexedSlices) else grad for grad in gradients]
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)).numpy() for grad in new_gradients))
    norm = tf.cast(norm, dtype=tf.float32)

    if tf.greater(norm, theta):
        new_gradients = map(lambda grad: grad * theta / norm, new_gradients)
    return new_gradients
