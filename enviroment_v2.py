import numpy as np

class Environment(object):
    def __init__(self):
        self.whoami = 'environment'
        self.reward_f = 'close'  # vanila, close
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
        self.stock_price = 0

        # self.fee = 0.0005
        self.fee = 0.0000

        self.data_structure = ''

        # without short option
        # self.scheme = [[0, 0, 0],  # when your stock = -1
        #                [0, 0, 1],    # when your stock = 0
        #                [0, 1, 1]]    # when your stock = 1

        # without short option,
        self.scheme = [[0, 0, 0],  # when your stock = -1
                       [0, 0, 100],    # when your stock = 0
                       [0, 100, 100]]    # when your stock = 1

        # with short option - extreme
        # self.scheme = [[-1, -1, 1],   # when your stock = -1
        #                [-1, 0, 1],    # when your stock = 0
        #                [-1, 1, 1]]    # when your stock = 1

        # with short option - moderate
        # self.scheme = [[-1, -1, 0],   # when your stock = -1
        #                [-1, 0, 1],    # when your stock = 0
        #                [0, 1, 1]]    # when your stock = 1


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

        else:
            self.data_structure = mode
            import csv
            with open("sberdata_leak.csv", newline='') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=";")

                for row in reader:
                    for value in row:
                        row[value] = row[value].replace(",", ".")
                        row[value] = row[value].replace(" ", "")
                    self.data.append(row)

        pass

    def getdata(self, row):

        if self.data_structure == 'leak' or self.data_structure == 'random':
            result = np.array(
                [float(self.data[row]['data-leak']) - float(
                    self.data[row]['price'])])  # let's take only the price change
        elif self.data_structure == 'oldprice':
            result = np.array(
                [float(self.data[row][
                           'price']) - self.stock_price])  # let's pass the diff between history price & current price
        else:
            result = np.array([])

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

        if self.stock == 0:
            new_stock = self.scheme[1][action + 1]
        elif self.stock > 0:
            new_stock = self.scheme[2][action + 1]
        elif self.stock < 0:
            new_stock = self.scheme[0][action + 1]

        diff_stock = new_stock - self.stock

        old_price = float(prev_state['price'])
        new_price = float(new_state['price'])

        # Расчет комиссии с транзакции
        if diff_stock != 0:
            trans_fee = abs(diff_stock) * float(old_price) * self.fee
            self.balance = self.balance - trans_fee - diff_stock * float(old_price)
            self.stock_price = float(old_price)

        end_flag = False

        if self.episode_step == self.episode_length:
            end_flag = True

        new_total = 0
        if self.reward_f == 'vanila':
            # Total и reward считаем уже по новой цене
            new_total = self.balance + new_stock * float(new_price)
        elif self.reward_f == 'close':
            # total считаем по цене покупки стока
            if end_flag:
                # в любом случае закрываем позицию в конце эпизода
                new_total = self.balance + new_stock * float(new_price)
            else:
                new_total = self.balance + new_stock * self.stock_price
        else:
            pass
        reward = new_total - self.total
        self.total = new_total
        self.stock = new_stock
        return_state = self.getdata(current_step)  # calculate return state using updated stock price

        return return_state, self.stock, reward, end_flag
