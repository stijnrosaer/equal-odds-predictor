import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self):
        self.A = random.randint(0, 1)
        self.R = random.random()

        self.Y = 0
        r = random.random()
        if self.A == 1:
            if self.R > 0.6:
                self.Y = random.randint(0,1)
        else:
            if self.R > 0.8:
                self.Y = random.randint(0,1)

    def __str__(self):
        return f"A: {self.A}, Y: {self.Y} R: {self.R}"

    def __repr__(self):
        return str(self)


def plot(data, sensitive_value):
    data_val = [i for i in data if i.A == sensitive_value]

    """
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
    """


    a,b,c = roc_curve([i.Y for i in data_val], [i.R for i in data_val])
    plt.plot(a, b)
    plt.plot([0,1], [0,1])
    plt.xlabel("False positive ratio")
    plt.ylabel("True positive ratio")
    # plt.xlim([0,1])
    # plt.ylim([0, 1])
    plt.show()



if __name__ == '__main__':

    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, weights=[0.5])

    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)

    model = LogisticRegression(solver="lbfgs")
    model.fit(trainX, trainy)
    yhat = model.predict_proba(testX)

    # keep probabilities for the positive outcome only
    yhat = yhat[:, 1]
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(testy, yhat)
    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))
    # locate the index of the largest g-mean
    # ix = argmax(gmeans)
    # print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    # plot the roc curve for the model
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='Logistic')
    # plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    # show the plot
    plt.show()


    # data = [Data() for _ in range(1000)]
    #
    # plot(data, 0)
    exit(0)
