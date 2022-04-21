import numpy as np
import tensorflow as tf
import keras_tuner as kt
from d2l import tensorflow as d2l
import tensorflow.keras as keras
from gradientclipper import gradient_clipping
import tf_slim as slim
import numpy as np
from functools import partial


class RNNModel(keras.models.Model):
    def __init__(self, vocab_size, num_hiddens, init_state, forward_fn, get_params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.init_state, self.forward_fn = init_state, forward_fn
        self.trainable_vars = get_params(vocab_size, num_hiddens)

    def __call__(self, X, state):
        X = tf.one_hot(tf.transpose(X), self.vocab_size)
        X = tf.cast(X, tf.float32)
        return self.forward_fn(X, state, self.trainable_vars)

    def begin_state(self, batch_size, *args, **kwargs):
        return self.init_state(batch_size, self.num_hiddens)


def get_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size
    with slim.arg_scope([slim.model_variable],
                        initializer=tf.random_normal_initializer(stddev=0.01, mean=0),
                        dtype=tf.float32, device=f'/GPU:0'):
        # hidden layer parameters
        w_xh = slim.model_variable('w_xh', shape=(num_inputs, num_hiddens))
        w_hh = slim.model_variable('w_hh', shape=(num_hiddens, num_hiddens))
        b_h = slim.model_variable('b_h', shape=num_hiddens, initializer=tf.zeros_initializer(), dtype=tf.float32)
        # output layer parameters
        w_hq = slim.model_variable('w_hq', shape=(num_hiddens, num_outputs))
        b_q = slim.model_variable('b_q', shape=num_outputs, initializer=tf.zeros_initializer(), dtype=tf.float32)

        params = [w_xh, w_hh, b_h, w_hq, b_q]
        return params


def init_rnn_state(batch_size, num_hiddens):
    return (tf.zeros((batch_size, num_hiddens)),)


def rnn(inputs, state, params):
    w_xh, w_hh, b_h, w_hq, b_q = params
    H, = state
    outputs = []

    for X in inputs:
        X = tf.reshape(X, [-1, w_xh.shape[0]])
        H = tf.tanh(tf.matmul(X, w_xh) + tf.matmul(H, w_hh) + b_h)
        Y = tf.matmul(H, w_hq) + b_q
        outputs.append(Y)
    return tf.concat(outputs, axis=0), (H,)


class RNNHyperModel(kt.HyperModel):
    def __init__(self, vocab_size, use_random_iter=False):
        self.vocab_size = vocab_size
        self.use_random_iter = use_random_iter

    def build(self, hp):
        num_hiddens = hp.Int(name='hidden_units', min_value=512, max_value=1024, step=128)
        return RNNModel(self.vocab_size, num_hiddens, init_rnn_state, rnn, get_params)

    def fit(self, hp, model, training_data, validation_data=None, callbacks=None, **kwargs):

        no_of_epochs = hp.Choice(name='epochs', values=[500, 1000])
        lr = hp.Choice('learning_rate', values=[1.0, 1e-1, 1e-2])

        # Define the optimizer.
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        updater = tf.keras.optimizers.SGD(lr)

        epoch_loss_metric = keras.metrics.Mean()

        # Function to run the train step.
        # @tf.function
        def run_train_step(train_dataset):
            state = None
            metric = d2l.Accumulator(2)
            for X, Y in train_dataset:
                if state is None or self.use_random_iter:
                    state = model.begin_state(batch_size=X.shape[0], dtype=tf.float32)
                with tf.GradientTape(persistent=True) as g:
                    y_hat, state = model(X, state)
                    y = tf.reshape(tf.transpose(Y), (-1))
                    l = loss(y, y_hat)
                params = model.trainable_vars
                gradients = g.gradient(l, params)
                gradients = gradient_clipping(gradients, 1)
                updater.apply_gradients(zip(gradients, params))
                metric.add(l * d2l.size(y), d2l.size(y))
            epoch_loss_metric.update_state(tf.exp(metric[0] / metric[1]))

        # Assign the model to the callbacks.
        for callback in callbacks:
            callback.model = model

        perplexity = float("inf")

        # The custom training loop.
        for epoch in range(no_of_epochs):
            # Iterate the training data to run the training step.
            run_train_step(training_data)

            # Calling the callbacks after epoch.
            ppl = float(epoch_loss_metric.result().numpy())
            for callback in callbacks:
                # The "perplexity" is the objective passed to the tuner.
                callback.on_epoch_end(epoch, logs={"perplexity": perplexity})
            epoch_loss_metric.reset_states()

            perplexity = min(ppl, perplexity)
            if(epoch % 100) == 0:
                print(f'perplexity {perplexity:.1f} after epoch {epoch}')
        # Return the evaluation metric value.
        print(f'perplexity {perplexity:.1f} for hyperparameters {hp.values}')
        return perplexity
