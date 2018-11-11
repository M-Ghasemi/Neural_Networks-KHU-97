import numpy as np


class Adaline(object):

    def __init__(self, eta=0.01, n_epoch=50, random_state=1):
        self.eta = eta
        self.n_epoch = n_epoch
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        augmented_X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        for _ in range(self.n_epoch):
            pred = augmented_X.dot(self.w_)
            errors = y - pred

            self.w_ += self.eta * augmented_X.T.dot(errors)

            cost = 1 / 2 * sum((errors ** 2))
            self.errors_.append(cost)

        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
