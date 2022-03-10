import tensorflow as tf
import tensorflow.keras as keras
from d2l import tensorflow as d2l

from vgg16 import vgg16

convo_arch = [(2, 64), (2, 128), (4, 256), (4, 512), (4, 512)]
net = vgg16(convo_arch)
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs, d2l.try_gpu()._device_name)
net.fit(train_iter, epochs=num_epochs, verbose=1, callbacks=[callback])
    

convo_arch = [(2, 64), (2, 128), (4, 256), (4, 512), (4, 512)]
net = vgg16(convo_arch)

    
net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs, d2l.try_gpu()._device_name)
net.fit(train_iter, epochs=num_epochs, verbose=1, callbacks=[callback])
   
