import numpy as np
# import enviroment as en
import enviroment_v2 as en
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
tf.compat.v1.disable_eager_execution()
disable_eager_execution()

epsilon_decay = 0.997
epsilon_bound = 0.1
epsilon = 1.0
M_episodes = 1000
epsilon_array = []

for i in range(M_episodes):
    epsilon *= epsilon_decay
    if epsilon < epsilon_bound:
        epsilon = epsilon_bound
    epsilon_array.append(epsilon)

plt.plot(epsilon_array)
plt.show()

# alpha = 0.001
#
# # create nn
# nn = tf.keras.Sequential([
#     tf.keras.layers.Dense(32, input_shape=(3,), activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(3, activation=None)
# ])
#
# # compile nn
# nn.compile(optimizer=tf.keras.optimizers.Adam(lr=alpha), loss='mse')
#
# state = np.array([100.0, 15.0, 0.0])
#
# # predict on batch
# value = nn.predict_on_batch(np.array([state]))
# print('value', value)
# action = np.argmax(value)
# print('action', action)
# # right answer
# reward = 50.0
# value[0][action] = reward
# print('updated value', value)
#
# # train on batch
# for i in range(1000):
#     nn.train_on_batch(np.array([state]), value)
#
# value_new = nn.predict_on_batch(np.array([state]))
# print('value_new', value_new)
