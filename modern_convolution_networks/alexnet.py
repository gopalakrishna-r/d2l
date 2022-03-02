import tensorflow as tf
from d2l import tensorflow as d2l
from NetworkUtils import alexnet

from tensorpack.tfutils.tower import TowerContext

input_shape = (224, 224, 1)

X = tf.random.uniform((1, 224, 224, 1))
net = alexnet(X.shape[1:], 10)
for layer in net.layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
with TowerContext('', is_training=True):
    lr, num_epochs = 0.01, 10
    d2l.train_ch6(net, train_iter, test_iter, batch_size, lr, num_epochs)
