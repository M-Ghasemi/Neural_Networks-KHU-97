import numpy as np


def perceptron_rule(X, Y, maximum_iteration=None, verbose=True):

    if maximum_iteration is None:
        maximum_iteration = X.shape[0] * 20

    # INITIAL WEIGHTS
    W = np.zeros(X.shape[1], dtype=int)

    # ITERATION
    t = 0
    # INPUT NUMBER
    k = 0
    # CHECK IF ALL PREDICTED VALUES ARE CORRECT (STOP CONDITION)
    correct_predicted = np.zeros_like(Y)

    # if x > 0 returns 1, elif x < 0 returns -1, else returns 0
    sign = lambda a: (a > 0) - (a < 0)

    # REPEAT UNTIL ALL OUTPUTS ARE EQUAL TO DESIRED OUTPUTS
    while not np.all(correct_predicted) and t < maximum_iteration:

        y = W.dot(X[k].T)

        if verbose:
            print(f"\nt: {t}, k: {k} \t W: {W}, X: {X[k]} \t y: {y}, desired: {Y[k]}")

        y = sign(int(y))

        if y == Y[k]:
            correct_predicted[k] = 1
        else:
            W += Y[k] * X[k]
            correct_predicted *= 0

        t += 1
        k = (k + 1) % X.shape[0]

    if verbose:
        print(f'Final W: {W}')

    return W


# INPUT
X = np.array([
    [1, -1, -1, -1],
    [1, 1, -1, -1],
    [1, 1, 1, 1]])
# DESIRED (REAL) OUTPUT
Y = np.array([1, -1, 1])

W = perceptron_rule(X, Y)


"""
import numpy as np

# INPUT
X = np.array([
    [1, -1, -1, -1],
    [1, 1, -1, -1],
    [1, 1, 1, 1]])
# DESIRED (REAL) OUTPUT
Y = np.array([1, -1, 1])

# INITIAL WEIGHTS
W = np.array([0, 0, 0, 0])

# ITERATION
t = 0
# INPUT NUMBER
k = 0
# CHECK IF ALL PREDICTED VALUES ARE CORRECT (STOP CONDITION)
correct_predicted = np.zeros_like(Y)

# REPEAT UNTIL ALL OUTPUTS ARE EQUAL TO DESIRED OUTPUTS
while not np.all(correct_predicted):

    y = int(W.dot(X[k].T))

    print(
        't: {}, k: {}\n'
        'W: {}, X: {}\n'
        'y: {}, desired: {}\n'.format(t, k, W, X[k], y, Y[k])
    )

    # if y>0 then y=1 (positive label), if y<0 then y=-1 (negative label), if y==0 then y does not change
    y = (y > 0) - (y < 0)

    if y == Y[k]:
        correct_predicted[k] = 1
    else:
        W += Y[k] * X[k]
        correct_predicted *= 0

    t += 1
    k = (k + 1) % X.shape[0]

print(W)
"""
