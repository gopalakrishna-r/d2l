import keras_tuner as kt
import matplotlib.pyplot as plt
import tensorflow as tf
from d2l import tensorflow as d2l

from model import RNNHyperModel
from parameters import get_params
from predictor import predict_ch8


batch_size = 32

for num_steps in range(35, 60, 5):
    num_hiddens = 512
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    tuner = kt.Hyperband(
        objective=kt.Objective('perplexity', direction='min'),
        directory="results",
        hypermodel=RNNHyperModel(len(vocab), use_random_iter=False),
        project_name="rnn",
        overwrite=True, seed=42)

    tuner.search(training_data=train_iter)

    best_hps = tuner.get_best_hyperparameters()[0]
    print(best_hps.values)

