import numpy as np

class Environment(object):
    def __init__(self):
        self.whoami = 'environment'
        self.data = []
        self.cur_ep = []
        self.array = []
        self.start_balance = 1000
        self.balance = 1000
        self.stock = 0

        # number of calculated additional columns
        self.derivative_columns = 1

        self.fee = 0.0005
        # Массив решений action -1 0 1 и stock -1 0 1
        self.scheme = [[-1, -1, 1], [-1, 0, 1], [-1, 1, 1]]

    def loaddata(self):
        columns = 0
        import csv
        with open("sberdata_leak.csv", newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=";")

            for row in reader:
                for value in row:
                    row[value] = row[value].replace(",", ".")
                    row[value] = row[value].replace(" ", "")
                if columns == 0:
                    columns = len(row)

                self.data.append(row)

        return columns - 1

    def load_episod(self, num=1):
        self.cur_ep = []

        for row in self.data:
            if int(row["episod"]) == num:
                self.cur_ep.append(row)

    def dict_to_list(self, array_dict):

        array_list = []

        for key in array_dict:
            array_list.append(float(array_dict[key]))

        array_list.pop(0)
        np_array = np.array(array_list)

        return np_array

    def start(self, episode_num):
        array = []
        self.balance = 1000
        self.stock = 0
        self.total = self.balance

        # Загружаем новый эпизод
        self.load_episod(episode_num)

        # На старте запускаем action с 0
        self.array = self.cur_ep.pop(0)
        array = self.dict_to_list(self.array)

        return array, [0]

    def action(self, action):
        # Новый остаток
        trans_fee = 0
        new_stock = self.scheme[self.stock + 1][action + 1]
        diff_stock = new_stock - self.stock

        old_price = self.array["price"]

        # Расчет комиссии с транзакции
        if diff_stock != 0:
            trans_fee = abs(diff_stock) * float(old_price) * self.fee
            self.balance = self.balance - trans_fee - diff_stock * float(old_price)

        end_flag = False
        self.array = self.cur_ep.pop(0)

        if len(self.cur_ep) == 0:
            end_flag = True

        # Total и reward считаем уже по новой цене
        new_total = self.balance + new_stock * float(self.array["price"])
        reward = new_total - self.total

        self.total = new_total
        self.stock = new_stock

        array = self.dict_to_list(self.array)
        derivative_array = np.array([new_stock])

        return array, derivative_array, reward, end_flag
