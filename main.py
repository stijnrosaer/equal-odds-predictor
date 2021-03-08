import numpy as np
from scipy.special import expit

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

x = np.arange(0, 10, 1).reshape(-1, 1)
y = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 1])

model = LogisticRegression(solver='liblinear')
model.fit(x, y)

x_test = np.arange(0, 10, 0.01)
y_test = x_test * model.coef_ + model.intercept_

sigmoid = expit(y_test)

y_test_val = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1])
x_test_val = [0, 1, 1.5, 3, 4, 5, 7, 8, 9]
color = ["blue", "blue", "blue", "green", "green", "red", "green", "green", "green"]


plt.scatter(x_test_val, y_test_val, c=color)

plt.plot(x_test, sigmoid.ravel(), label="logistic fit")
plt.hlines(0.5, 0, 10, linestyles={'dashed'}, color="gray", label="treshold= 0.5 ")
plt.vlines(2.1, 0, 1, linestyles={'dotted'}, color="gray")
plt.scatter(5,0, c="red", label="misclassified")

plt.ylabel("probability of obesity")
plt.xlabel("weight of animal")
plt.legend(loc="lower right")
plt.show()



# a-conditional
y_m = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1])
x_m = np.array([0, 1, 1.5, 3, 4, 5, 6, 7, 9])*1.2
color_m = ["blue" for _ in range(len(y_test_val))]

y_f = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1])
x_f = [0, 1, 1.5, 3, 4, 5, 7, 8, 9]
color_f = ["red" for _ in range(len(y_test_val))]


plt.scatter(x_m, y_m, c=color_m, label="male weight")
plt.scatter(x_f, y_f, c=color_f, label="female weight")

plt.plot(x_test, sigmoid.ravel(), label="logistic fit")
plt.hlines(0.5, 0, 10, linestyles={'dashed'}, color="gray", label="treshold= 0.5 ")
plt.vlines(2.1, 0, 1, linestyles={'dotted'}, color="gray")

plt.ylabel("probability of obesity")
plt.xlabel("weight of animal")
plt.legend(loc="lower right")
plt.show()

#a-cond ROC
fpr_m, tpr_m, thresholds_m = roc_curve(y_m, list(x_m), drop_intermediate=False)

plt.plot(fpr_m, tpr_m, label="male", c="blue")
# for i in range(len(fpr_m)):
#     plt.annotate(round(thresholds_m[i], 2), (fpr_m[i]+0.01, tpr_m[i]+0.01))

fpr_f, tpr_f, thresholds_f = roc_curve(y_f, x_f, drop_intermediate=False)

plt.plot(fpr_f, tpr_f, label="female", c="red")
# for i in range(len(fpr_f)):
#     plt.annotate(round(thresholds_f[i], 2), (fpr_f[i]+0.01, tpr_f[i]+0.01))

plt.xlabel("False positive rate (FPR)")
plt.ylabel("True positive rate (TPR)")
plt.title("A-conditional ROC curve")
plt.legend(loc="lower right")
plt.show()

#ROC
join_y = [*y_m , *y_f]
join_x = [*x_m , *x_f]
fpr, tpr, thresholds = roc_curve(join_y, join_x, drop_intermediate=False)

plt.plot(fpr, tpr)
for i in range(len(fpr)):
    plt.annotate(round(thresholds[i],2), (fpr[i]+0.01, tpr[i]+0.01))
plt.fill_between(fpr_1.sample(1000), (tpr_1.sample(1000)), fpr_2.sample(1000),tpr_2.sample(1000), alpha=0.3)
plt.legend()
plt.xlabel("False positive rate (FPR)")
plt.ylabel("True positive rate (TPR)")
plt.title("ROC curve")
plt.show()