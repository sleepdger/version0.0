import numpy as np

class Environment(object):
    def __init__(self):
        self.whoami = 'environment'
        self.data = []
        self.episode_length = 1000
        self.data_length = int(100 * 1.2 * self.episode_length)
        self.episode_step = 0
        self.episode_num = 0

        self.start_balance = 1000
        self.start_stock = 0

        self.balance = 1000
        self.total = 1000
        self.stock = 0

        self.fee = 0.0005

        # Массив решений action и stock
        self.scheme = [[-1, -1, 1],  # stock = -1
                       [0, 0, 1],  # stock = 0
                       [0, 1, 1]]  # stock = 1

    def loaddata(self):

        array = np.random.rand(2, self.data_length)
        array[0] = array[0] * 100.0 + 200.0
        array[1] = 0.0
        array = np.round(np.transpose(array), 2)

        i = 0
        while i + 1 < len(array):
            array[i][1] = array[i + 1][0] - array[i][0]
            i += 1

        self.data = np.copy(array)

        return len(array[0]) + 1

    def start(self, episode_num):

        self.total = self.start_balance
        self.balance = self.start_balance
        self.stock = self.start_stock
        self.episode_step = 0
        self.episode_num = episode_num

        return self.data[episode_num * self.episode_length], self.stock

    def action(self, action):

        self.episode_step += 1
        current_step = self.episode_num * self.episode_length + self.episode_step
        prev_state = self.data[current_step - 1]
        new_state = self.data[current_step]

        new_stock = self.scheme[self.stock + 1][action + 1]
        diff_stock = new_stock - self.stock

        old_price = prev_state[0]
        new_price = new_state[0]

        # Расчет комиссии с транзакции
        if diff_stock != 0:
            trans_fee = abs(diff_stock) * float(old_price) * self.fee
            self.balance = self.balance - trans_fee - diff_stock * float(old_price)

        end_flag = False

        if self.episode_step == self.episode_length:
            end_flag = True

        # Total и reward считаем уже по новой цене
        new_total = self.balance + new_stock * float(new_price)
        reward = new_total - self.total

        self.total = new_total
        self.stock = new_stock

        return new_state, self.stock, reward, end_flag
