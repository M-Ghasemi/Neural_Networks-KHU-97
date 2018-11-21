import numpy as np


class Adaline(object):
    """Adaline classifier

    Attributes:
        eta (float): the learning rate.
        epoch (int): the number of epochs.
    """

    def __init__(self, eta=0.001, epoch=100):
        self.eta = eta
        self.epoch = epoch

    def fit(self, X, y):
        """Fits the training data.

        Args:
            X : array-like (training vectors) of shape [n_samples, n_features],
                where n_samples is the number of samples and n_features is the
                number of features.
            y : array-like (Target values) of shape [n_samples].

        Returns:
            self : object
        """
        np.random.seed(16)
        self.weight_ = np.random.uniform(-1, 1, X.shape[1] + 1)
        self.error_ = []

        for _ in range(self.epoch):

            output = self.activation_function(X)
            error = y - output

            self.weight_[0] += self.eta * sum(error)
            self.weight_[1:] += self.eta * X.T.dot(error)

            cost = (1 / 2) * sum((error**2))
            self.error_.append(cost)

        return self

    def net_input(self, X):
        """Calculate the net input z"""
        return np.dot(X, self.weight_[1:]) + self.weight_[0]
    def activation_function(self, X):
        """Calculate the output g(z)"""
        return self.net_input(X)
    def predict(self, X):
        """Return the binary value 0 or 1"""
        return np.where(self.activation_function(X) >= 0.0, 1, -1)
