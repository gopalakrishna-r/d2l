import tensorflow as tf
from d2l import tensorflow as d2l

mse_loss = tf.keras.losses.MeanSquaredError()

def get_net():
    net = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='relu'),
                              tf.keras.layers.Dense(1)])
    return net


def compute_loss_grads(m_input, y, net):
    with tf.GradientTape() as tape:
        y_predictions = net(m_input)
        loss = mse_loss(y, y_predictions)
    grads = tape.gradient(loss, net.trainable_vars)
    return loss, grads


def train(net, train_iter, epochs):
    trainer = tf.keras.optimizers.Adam()
    for epoch in range(epochs):
        for X, y in train_iter:
            _, grads = compute_loss_grads(X, y, net)
            trainer.apply_gradients(zip(grads, net.trainable_vars))
        print(f'epoch {epoch + 1}, '
              f'loss : {d2l.evaluate_loss(net, train_iter, mse_loss):f}')


