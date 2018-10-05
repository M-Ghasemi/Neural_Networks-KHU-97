
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

# if x > 0 returns 1, elif x < 0 returns -1, else returns 0
sign = lambda a: (a > 0) - (a < 0)

# REPEAT UNTIL ALL OUTPUTS ARE EQUAL TO DESIRED OUTPUTS
while not np.all(correct_predicted):

    y = W.dot(X[k].T)

    print(
        't: {}, k: {}\n'
        'W: {}, X: {}\n'
        'y: {}, desired: {}\n'.format(t, k, W, X[k], y, Y[k])
    )

    # y = -1 if y < 0 else 1
    # if y < 0:
    #     y = -1
    # elif y > 0:
    #     y = 1
    # else:
    #     y = Y[k] * -1  # IF y == 0, THEN PREDICTED VALUE IS FALSE (WEIRD)
    y = sign(int(y))

    if y == Y[k]:
        correct_predicted[k] = 1
    else:
        W += Y[k] * X[k]
        correct_predicted *= 0

    t += 1
    k = (k + 1) % X.shape[0]

print(W)
