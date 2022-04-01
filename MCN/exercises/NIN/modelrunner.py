import keras_tuner as kt
import tensorflow as tf
from d2l import tensorflow as d2l

from netowrkinnetwork import build_graph

num_epochs, batch_size =  10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
tuner = kt.Hyperband(
    build_graph,
    objective=kt.Objective('accuracy', direction='max'),max_epochs=20, factor=2,directory="results_dir",
    project_name="mnist",  overwrite=True, seed=42)

print(f'search summary /t {tuner.search_space_summary()}')

tuner.search(train_iter, epochs = 20, workers = 8, use_multiprocessing=True, callbacks = [tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)])

print(tuner.get_best_hyperparameters()[0].values)

net =  tuner.get_best_models()[0]

callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs, d2l.try_gpu()._device_name)
net.fit(train_iter, epochs=num_epochs, verbose=1, callbacks=[callback])

net.evaluate(test_iter)

   
