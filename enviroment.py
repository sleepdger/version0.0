import numpy as np

class Environment(object):
    def __init__(self):
        self.whoami = 'environment'
        self.data = []
        self.cur_ep = []
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
        with open("sberdata.csv", newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=";")

            for row in reader:
                for value in row:
                    row[value] = row[value].replace(",",".")
                    row[value] = row[value].replace(" ","")
                if columns == 0:
                    columns = len(row)

                self.data.append(row)

        return columns-1

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
        self.start_balance = 1000
        self.balance = 1000
        self.stock = 0

        # Загружаем новый эпизод
        self.load_episod(episode_num)

        # На старте запускаем action с 0
        array, derivative_array, reward, end_flag = self.action(0)

        return array, derivative_array

    def action(self, action):
        end_flag = False
        array = self.cur_ep.pop(0)

        if len(self.cur_ep) == 0:
            end_flag = True

        # Новый остаток
        if self.stock > 1 or action > 1:
            print(self.cur_ep)
        new_stock = self.scheme[self.stock + 1][action + 1]
        diff_stock = new_stock - self.stock

        # Расчет комиссии с транзакции
        if new_stock != 0:
            trans_fee = abs(diff_stock) * float(array["price"]) * self.fee
            self.balance = self.balance - trans_fee - diff_stock * float(array["price"])

        reward = self.balance + new_stock * float(array["price"]) - self.start_balance

        self.stock = new_stock

        array = self.dict_to_list(array)
        derivative_array = np.array([new_stock])

        return array, derivative_array, reward, end_flag

#
# env = Environment()
# # Получаем данные
# env.loaddata()
#
# start = env.start(1)
#
# print(start)
# #Запускаем Action
#
# action = env.action(1)
# print(action)
# print(env.balance)
# print(env.stock)
# action = env.action(0)
# print(action)
# print(env.balance)
# print(env.stock)
# action = env.action(1)
# print(action)
# print(env.balance)
# print(env.stock)
# action = env.action(-1)
# print(action)
# print(env.balance)
# print(env.stock)
# action = env.action(-1)
# print(action)
# print(env.balance)
# print(env.stock)
# action = env.action(0)
# print(action)
# print(env.balance)
# print(env.stock)
# action = env.action(0)
# print(action)
# print(env.balance)
# print(env.stock)
# action = env.action(0)
# print(action)
# print(env.balance)
# print(env.stock)