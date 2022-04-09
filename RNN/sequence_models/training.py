import tensorflow as tf
from d2l import tensorflow as d2l
import matplotlib.pyplot as plt
from model_fns import get_net, train, mse_loss

T = 1000
time = tf.range(1, T + 1, dtype=tf.float32)
X = tf.sin(0.01 * time) + tf.random.normal([T], 0, 0.2)
d2l.plot(time, [X], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))

#plt.show(block=True)

tau = 4
features = tf.Variable(tf.zeros((T - tau, tau)))
for i in range(tau):
    features[:, i].assign(X[i:T - tau + i])
labels = tf.reshape(X[tau:], (-1, 1))
batch_size, n_train = 16, 600
train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)

net = get_net()
train(net, train_iter, 5, 0.01)

one_step_prediction = net(features)
d2l.plot([time, time[tau:]], [X.numpy(), one_step_prediction.numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3))
# plt.show()

# multistep ahead prediction
multistep_predictions = tf.Variable(tf.zeros(T))
multistep_predictions[:n_train + tau].assign(X[:n_train + tau])
for i in range(n_train + tau, T):
    multistep_predictions[i].assign(tf.reshape(net(tf.reshape(multistep_predictions[i - tau:i], (1, -1))), ()))

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [X.numpy(), one_step_prediction.numpy(), multistep_predictions[n_train + tau:].numpy()], 'time',
         'x', legend=['data', '1-step precisions', 'multistep predictions'],
         xlim=[1, 1000], figsize=(6, 3))


# kstep ahead predictions
max_steps = 64

features = tf.Variable(tf.zeros((T-tau-max_steps +1, tau+max_steps)))

for i in range(tau):
    features[:, i].assign(X[i:i+T-tau-max_steps+1].numpy())

for i in range(tau, tau+max_steps):
    features[:, i].assign(tf.reshape(net((features[:, i-tau:i])), -1))

steps = (1, 4, 16, 64)
d2l.plot([time[tau+i-1:T-max_steps+i] for i in steps],
         [features[:, (tau + i -1)].numpy() for i in steps],
         'time', 'x', legend= [f'{i}-step preds' for i in steps],
         xlim = [5, 1000], figsize=(6, 3))
plt.show()

