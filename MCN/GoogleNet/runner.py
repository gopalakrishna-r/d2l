import tensorflow as tf
from d2l import tensorflow as d2l

from InceptionStem import build_graph

net = build_graph()

X = tf.random.uniform((1, 96, 96, 1))
for layer in net.layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
    
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs, d2l.try_gpu())
net.fit(train_iter, epochs=num_epochs, verbose=1, callbacks=[callback])