import numpy as np

class Environment(object):
    def __init__(self):
        self.whoami = 'environment'
        self.data = []
        self.episode_length = 1000
        self.data_length = int(1000 * 1.2 * self.episode_length)
        self.episode_step = 0
        self.episode_num = 0

        self.start_balance = 1000
        self.start_stock = 0

        self.balance = 1000
        self.total = 1000
        self.stock = 0

        # self.fee = 0.0005
        self.fee = 0.0000

        self.data_structure = 'random'

        # without short option
        # self.scheme = [[-1, -1, 1],  # when your stock = -1
        #                [0, 0, 1],    # when your stock = 0
        #                [0, 1, 1]]    # when your stock = 1

        # with short option
        self.scheme = [[-1, -1, 1],   # when your stock = -1
                       [-1, 0, 1],    # when your stock = 0
                       [-1, 1, 1]]    # when your stock = 1


    def loaddata(self, mode):

        # episod
        # price
        # Volume
        # VolumeData
        # VolumeDeltaCum
        # TradesCount
        # TradesCountDelta
        # TradesCountDeltaCum
        # IMOEX.price
        # data-leak

        if mode == 'random':
            self.data_structure = 'random'
            array = np.random.rand(2, self.data_length)
            array[0] = array[0] * 100.0 + 200.0
            array[1] = 0.0
            array = np.round(np.transpose(array), 2)

            i = 0
            while i + 1 < len(array):
                self.data.append({'price': array[i][0], 'data-leak': array[i + 1][0]})
                i += 1

        elif mode == 'leak':
            self.data_structure = 'leak'
            import csv
            with open("sberdata_leak.csv", newline='') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=";")

                for row in reader:
                    for value in row:
                        row[value] = row[value].replace(",", ".")
                        row[value] = row[value].replace(" ", "")
                    self.data.append(row)
        else:
            pass

        pass

    def getdata(self, row):

        result = np.array(
            [float(self.data[row]['data-leak']) - float(self.data[row]['price'])])  # let's take only the price change

        return result

    def start(self, episode_num):

        self.total = self.start_balance
        self.balance = self.start_balance
        self.stock = self.start_stock
        self.episode_step = 0
        self.episode_num = episode_num
        return_state = self.getdata(episode_num * self.episode_length)

        return return_state, self.stock

    def action(self, action):

        self.episode_step += 1
        current_step = self.episode_num * self.episode_length + self.episode_step
        prev_state = self.data[current_step - 1]
        new_state = self.data[current_step]
        return_state = self.getdata(current_step)

        new_stock = self.scheme[self.stock + 1][action + 1]
        diff_stock = new_stock - self.stock

        old_price = float(prev_state['price'])
        new_price = float(new_state['price'])

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

        return return_state, self.stock, reward, end_flag
