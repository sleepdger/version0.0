import numpy as np
import enviroment as en
import math
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
tf.compat.v1.disable_eager_execution()
disable_eager_execution()

class Agent(object):
    def __init__(self):
        self.whoami = 'agent'
        self.env = en.Environment()
        self.columns = self.env.loaddata() + self.env.derivative_columns

        # general parameters
        self.ep_step_cap = 1000
        self.debug = True
        self.num_actions = 3  # ['Sell', 'Do nothing', 'Buy']

        self.rewards = []
        self.rewards_100 = []
        self.rewards_100_avg = []
        self.trained_rewards = []
        self.D_memory = []

        # parameters
        self.gamma = 0.1  # 0.99
        self.epsilon = 1.0  # 1.0

        self.M_episodes = 1000
        self.T_episodes = 100

        # learned parameters
        self.minibatch = 32  # 32
        self.epsilon_decay = 0.995  # 0.997
        self.epsilon_bound = 0.1  # 0.1
        self.C_step = 1  # 200

        # tuning parameters
        self.alpha = 0.005  # 0.001
        self.D_N = 40000  # 40000

        # Initialize action-value function Q
        self.Q_nn = self.dqn_nn()
        # Initialize target action-value function ^Q
        self.Q_target_nn = self.dqn_nn()



    def dqn_nn(self):
        # create nn
        nn = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_shape=(self.columns,), activation='relu'),
            tf.keras.layers.Dense(64, input_shape=(self.columns,), activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(3, activation=None)
        ])

        # compile nn
        nn.compile(optimizer=tf.keras.optimizers.Adam(lr=self.alpha), loss='mse')

        return nn

    def draw_rewards(self, rewards, episodes, filename, title, flag, average):
        x = np.arange(0, episodes, 1)
        what_to_draw = len(rewards) - episodes
        rewards_to_draw = list(rewards[what_to_draw:])

        avg_reward = str(round(sum(rewards) / episodes, 1))
        filename_string = "A_" + str(self.alpha) + "_ED_" + str(self.epsilon_decay) + "_EB_" + str(
            self.epsilon_bound) + "_B_" + str(self.minibatch) + "_C_" + str(self.C_step) + "_D_" + str(
            self.D_N)

        plt.plot(x, rewards_to_draw)

        if not flag:
            plt.plot(x, average)

        plt.xlabel(filename_string)
        plt.ylabel('reward')
        title_string = title
        if flag:
            title_string += " (avg.reward: " + avg_reward + ")"
        plt.title(title_string)
        plt.grid(True)

        plt.savefig(filename_string + "_" + filename)
        plt.show()

    def training(self):

        episode = 1
        c = 0
        epsilon_decay_flag = True

        while episode <= self.M_episodes:

            first_state, derivative_state = self.env.start(episode)
            this_state = np.concatenate((first_state, derivative_state))

            end_flag = False
            t = 0
            total_reward = 0.0
            len_D_flag = False
            len_D = 0
            action_stats = [0,0,0]

            # run episode
            while (t < self.ep_step_cap) and not end_flag:
                t += 1

                # With probability e select a random action
                my_r = np.random.random()
                if my_r < self.epsilon:
                    action = np.random.randint(-1, 2)
                else:
                    Q_value = self.Q_nn.predict_on_batch(np.array([this_state]))
                    # convert action range (0, 1, 2) to (-1, 0, 1)
                    action = np.argmax(Q_value[0]) - 1

                action_stats[action+1] += 1

                # Execute action a_t in emulator and observe reward r_t and image x_t+1
                first_state, derivative_state, reward, end_flag = self.env.action(action)
                next_state = np.concatenate((first_state, derivative_state))

                total_reward += reward

                # store transition in D memory
                self.D_memory.append([reward, action, this_state, end_flag, next_state])
                this_state = next_state

                if not len_D_flag:
                    len_D = len(self.D_memory)
                    if len_D > self.D_N:
                        len_D = self.D_N
                        len_D_flag = True
                        self.D_memory.pop(0)
                    else:
                        # don't start training until we have full D memory
                        do = 'nothing'
                        # continue
                else:
                    # keep first memories
                    # self.D_memory.pop(self.D_N // 2)
                    self.D_memory.pop(0)

                # wait until we have some transitions in D memory
                if len_D < self.minibatch:
                    continue

                # Sample random mini-batch of transitions from D
                batch = random.sample(self.D_memory, self.minibatch)

                # list of states from the batch
                this_state_batch = np.array([i[2] for i in batch])
                next_state_batch = np.array([i[4] for i in batch])

                # get list of predictions for next states
                prediction_next_batch = self.Q_target_nn.predict_on_batch(next_state_batch)
                prediction_this_batch = self.Q_nn.predict_on_batch(this_state_batch)

                values_batch = []

                for i in range(len(batch)):

                    b_reward = batch[i][0]
                    b_action = batch[i][1]
                    b_end_flag = batch[i][3]

                    y_value = b_reward
                    if not b_end_flag:
                        y_value += self.gamma * np.max(prediction_next_batch[i])
                    q_value = prediction_this_batch[i]
                    q_value[b_action] = y_value

                    values_batch.append(q_value)

                # Perform a gradient descent step
                values_batch_np = np.array([i for i in values_batch])
                self.Q_nn.train_on_batch(this_state_batch, values_batch_np)

                # Every C steps reset target weights
                c += 1
                if c >= self.C_step:
                    c = 0
                    self.Q_target_nn.set_weights(self.Q_nn.get_weights())

            # epsilon decay after episode is over
            if epsilon_decay_flag:
                self.epsilon = self.epsilon * self.epsilon_decay
                if self.epsilon < self.epsilon_bound:
                    self.epsilon = self.epsilon_bound
                    epsilon_decay_flag = False

            # collecting statistics about finished episode
            self.rewards.append(total_reward)
            self.rewards_100.append(total_reward)
            if len(self.rewards_100) > 100:
                self.rewards_100.pop(0)
            self.rewards_100_avg.append(sum(self.rewards_100) / 100)

            # printing statistics about finished episode
            if self.debug:
                # self.env.render()
                print('total_reward', total_reward, 'episode', episode, 'Stats: buy', action_stats[2], 'hold', action_stats[1], 'sell', action_stats[0], 'total', sum(action_stats))

            # increasing episode counter
            episode += 1

    def testing(self):
        # now let's see what we've learned

        for q in range(self.T_episodes):
            first_state, derivative_state = self.env.start(self.M_episodes + 1 + q)
            state = np.concatenate((first_state, derivative_state))
            i = 0
            end_flag = False
            t_reward = 0
            action_stats = [0, 0, 0]

            while (i < self.ep_step_cap) and not end_flag:
                i += 1

                # choosing action according to learned Q function
                Q_value = self.Q_nn.predict_on_batch(np.array([state]))
                action = np.argmax(Q_value[0]) - 1

                # perform action
                first_state, derivative_state, reward, end_flag = self.env.action(action)
                state = np.concatenate((first_state, derivative_state))

                action_stats[action+1] += 1
                t_reward += reward

            if self.debug:
                print('t_reward', t_reward, 'episode', q, 'Stats: buy', action_stats[2], 'hold', action_stats[1], 'sell', action_stats[0], 'total', sum(action_stats))

            self.trained_rewards.append(t_reward)

my_agent = Agent()
my_agent.training()
my_agent.draw_rewards(my_agent.rewards, my_agent.M_episodes, "training_tr.png", 'Training', False,
                      my_agent.rewards_100_avg)
my_agent.testing()
my_agent.draw_rewards(my_agent.trained_rewards, my_agent.T_episodes, "running_tr.png", 'Trained agent', True, [])