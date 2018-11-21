import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
plt.scatter(X[:50, 0], X[:50, 1],color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()


perceptron = Perceptron()

errors = list()
for i in range(1, 300):
    perceptron.n_iter = i
    perceptron.fit(X, y)
    errors.append(sum(perceptron.predict(X) != y))


# DECISION BOUNDARY
SEPAL = [round(min(X[:, 0]) - 1), round(max(X[:, 0]) + 1)]
PETAL = [-((W[1] * sep) + W[0]) / W[2] for sep in SEPAL]

plt.plot(SEPAL, PETAL, 'r--')
