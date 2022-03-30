import tensorflow as tf
from d2l import tensorflow as d2l
from Inception import build_graph, net

net = net()

X = tf.random.uniform((1, 224, 224, 1))
for layer in net.layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)


lr, num_epochs, batch_size = 0.0015, 10, 128
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=lr,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs, d2l.try_gpu())
net.fit(train_iter, epochs=num_epochs, verbose=1, callbacks=[callback])