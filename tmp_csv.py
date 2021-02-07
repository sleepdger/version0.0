import numpy as np
import random
import matplotlib.pyplot as plt


# data_length = 100
# data = []
# array = np.random.rand(2, data_length)
# array[0] = array[0] * 100.0 + 200.0
# array[1] = 0.0
# array = np.round(np.transpose(array), 2)
#
# i = 0
# while i + 1 < len(array):
#     array[i][1] = array[i + 1][0] - array[i][0]
#     data.append({'episode': i, 'price': array[i][0], 'data-leak': array[i][1]})
#     i += 1

data = []
import csv

with open("sberdata_leak.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=";")

    for row in reader:
        for value in row:
            row[value] = row[value].replace(",", ".")
            row[value] = row[value].replace(" ", "")
        data.append(row)

print(data[50])
