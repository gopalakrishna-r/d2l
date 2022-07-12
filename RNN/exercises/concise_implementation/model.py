import tensorflow as tf
import tensorflow.keras as keras
import keras_tuner as kt
from d2l import tensorflow as d2l


class RNNOutputLayer(keras.layers.Layer):
    def __init__(self, rnn, output_vocab_size, **kwargs):
        super(RNNOutputLayer, self).__init__(**kwargs)
        self.rnn_layer = rnn
        self.vocab_size = output_vocab_size
        self.dense = keras.layers.Dense(output_vocab_size)

    def call(self, inputs, state):
        X = tf.one_hot(tf.transpose(inputs), self.vocab_size)
        Y, *state = self.rnn_layer(X, state)
        output = self.dense(tf.reshape(Y, [-1, Y.shape[-1]]))
        return output, state


class RNNCompositeModel(keras.models.Model):
    def __init__(self, num_hidden_state, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.rnn_cell = keras.layers.SimpleRNNCell(num_hidden_state,
                                                   kernel_initializer=keras.initializers.glorot_uniform)
        self.rnn_layer = keras.layers.RNN(self.rnn_cell, time_major=True,
                                          return_sequences=True, return_state=True)
        self.output_layer = RNNOutputLayer(self.rnn_layer, vocab_size)

    def call(self, inputs, state):
        return self.output_layer(inputs, state)

    def begin_state(self, *args, **kwargs):
        return self.rnn_layer.cell.get_initial_state(*args, **kwargs)


class RNNHyperModel(kt.HyperModel):
    def __init__(self, rnn_vocab_size):
        super().__init__()
        self.vocab_size = rnn_vocab_size

    def build(self, hp):
        num_hidden_state = hp.Int(name='hidden_units', min_value=256, max_value=1024, step=128)
        return RNNCompositeModel(num_hidden_state, self.vocab_size)

    def fit(self, hp, model, training_data, validation_data=None, callbacks=None, **kwargs):
        no_of_epochs = hp.Choice(name='epochs', values=[500])
        lr = hp.Choice('learning_rate', values=[1.0])

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
                if state is None:
                    state = model.begin_state(batch_size=X.shape[0], dtype=tf.float32)
                with tf.GradientTape(persistent=True) as g:
                    y_hat, state = model(X, state)
                    y = tf.reshape(tf.transpose(Y), (-1))
                    l = loss(y, y_hat)
                params = model.trainable_vars
                gradients = g.gradient(l, params)
                gradients = d2l.grad_clipping(gradients, 1)
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
            if (epoch % 100) == 0:
                print(f'perplexity {perplexity:.1f} after epoch {epoch}')
        # Return the evaluation metric value.
        print(f'perplexity {perplexity:.1f} for hyperparameters {hp.values}')
        return perplexity
