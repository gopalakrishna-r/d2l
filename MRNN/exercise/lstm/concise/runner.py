import tensorflow as tf
from d2l import tensorflow as d2l

import wandb


# Adjust the hyperparameters and analyze their influence on
# running time, perplexity, and the output sequence.


def build_rnn_model(config):
    lstm_cell = tf.keras.layers.LSTMCell(config.hidden_units, kernel_initializer="glorot_uniform")
    lstm_layer = tf.keras.layers.RNN(lstm_cell, time_major=True, return_sequences=True, return_state=True)
    return d2l.RNNModel(lstm_layer, len(vocab))


gpu_device_name = d2l.try_gpu()._device_name
with wandb.init(project="hyperparameter-sweeps-lstm", name='lstm-rnn', entity='goofygc316') as run:
    strategy = tf.distribute.OneDeviceStrategy(gpu_device_name)
    num_epochs = 500
    config = run.config
    batch_size, num_steps = config.batch_size, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    vocab_size = len(vocab)
    net = build_rnn_model(config)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    updater = tf.keras.optimizers.SGD(config.learning_rate)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = d2l.train_epoch_ch8(net, train_iter, loss, updater,
                                         use_random_iter=False)
        if (epoch + 1) % 10 == 0:
            wandb.log({
                "Epoch": epoch,
                "perplexity": ppl})
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(gpu_device_name)}')
