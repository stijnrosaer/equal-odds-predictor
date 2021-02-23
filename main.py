import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


class Data:
    def __init__(self):
        self.A = random.randint(0, 1)
        self.R = random.random()
        self.Y = random.randint(0, 1)

    def __str__(self):
        return f"A: {self.A}, Y: {self.Y} R: {self.R}"

    def __repr__(self):
        return str(self)


def plot(data, sensitive_value):
    data_val = [i for i in data if i.A == sensitive_value]
    # data_val.sort(key=lambda x: x.Y)


    fpr_tpr = []
    total_pos = len([i for i in data_val if i.Y == 1])
    total_neg = len([i for i in data_val if i.Y == 0])

    for threshold in range(len(data_val)):
        tp = fp = tn = fn = 0
        for item in data_val[:threshold]:  # ^y = 0
            if item.Y == 1:
                fp += 1
            else:
                tn += 1
        for item in data_val[threshold:]:  # ^y = 1
            if item.Y == 1:
                tp += 1
            else:
                fn += 1

        try:
            tpr = tp/(tp + fn)
        except:
            tpr = 0
        try:
            fpr = fp/(tn + fp)
        except:
            fpr = 0
        fpr_tpr.append((fpr, tpr))

    a, b, c = roc_curve(np.array([i.Y for i in data_val]), np.array([i.R for i in data_val]))

    plt.plot(a, b, lw=2)
    plt.xlabel("False positive ratio")
    plt.ylabel("True positive ratio")
    # plt.xlim([0,1])
    # plt.ylim([0, 1])
    plt.show()



if __name__ == '__main__':
    data = [Data() for _ in range(2000)]

    plot(data, 0)

    exit(0)
