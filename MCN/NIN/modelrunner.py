import tensorflow as tf
from netowrkinnetwork import build_graph
from d2l import tensorflow as d2l

net = build_graph()
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs, d2l.try_gpu()._device_name)
net.fit(train_iter, epochs=num_epochs, verbose=1, callbacks=[callback])

   