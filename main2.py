import numpy as np
from scipy.special import expit

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

x = np.arange(0, 10, 1).reshape(-1, 1)
y = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 1])

model = LogisticRegression(solver='liblinear')
model.fit(x, y)

x_test = np.arange(0, 10, 0.01)
y_test = x_test * model.coef_ + model.intercept_

sigmoid = expit(y_test)

x_test_val = [0, 1, 1.5, 3, 4, 5, 7, 7.5, 8, 9]
color = ["blue", "blue", "blue", "green", "green", "red", "green", "green", "green", "green"]
plt.scatter(x_test_val, y, c=color)

plt.plot(x_test, sigmoid.ravel(), label="logistic fit")
plt.hlines(0.5, 0, 10, linestyles={'dashed'}, color="gray", label="treshold= 0.5 ")
plt.vlines(2.1, 0, 1, linestyles={'dotted'}, color="gray")
plt.scatter(5,0, c="red", label="misclassified")

plt.ylabel("probability of obesity")
plt.xlabel("weight of animal")
plt.legend(loc="lower right")
plt.show()
