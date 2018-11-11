import numpy as np


class Perceptron(object):
    """Perceptron classifier.
    Parameters
    ------------
    eta : float
    Learning rate (between 0.0 and 1.0)
    n_iter : int
    Passes over the training dataset.
    random_state : int
    Random number generator seed for random weight
    initialization.
    Attributes
    -----------
    w_ : 1d-array
    Weights after fitting.
    errors_ : list
    Number of misclassifications (updates) in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of
        samples and n_features is the number of features.
        y : array-like, shape = [n_samples]
        Target values.
        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        # YOUR CODE HERE
        #
        # ADDING A 1 TO EACH ROW FOR BIAS
        augmented_X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        # ITERATION
        t = 0
        # INPUT NUMBER
        k = 0

        Predicted = np.zeros_like(y, dtype=int)
        # if x > 0 returns 1, elif x < 0 returns -1, else returns 0
        sign = lambda a: (a > 0) - (a < 0)

        # REPEAT UNTIL ALL OUTPUTS ARE EQUAL TO DESIRED OUTPUTS
        while t < self.n_iter:  # and not np.all(Predicted == y)

            pred = sign(int(augmented_X[k].dot(self.w_)))

            if pred == y[k]:
                Predicted[k] = pred
            else:
                self.w_ += y[k] * augmented_X[k]
                Predicted *= 0  # BECAUSE OF WEIGHT'S CHANGE WE SHOULD CHECK AGAIN LATER

            t += 1
            k = (k + 1) % augmented_X.shape[0]
        #
        # END YOUR CODE

        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
