import numpy as np


class LinearRegression:
    """
    w: np.ndarray
    b: float
    """

    def __init__(self):
        self.w = 0
        self.b = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """This function fits the linear regression model using the predictors X and the outcome y.

        Args:
            X (np.ndarray): m by n predictors matrix containing m samples and n features
            y (np.ndarray): m by 1 predicted outcome matrix
        """
        self.n_samples, self.n_feats = X.shape
        X = np.c_[X, np.ones(self.n_samples)]
        theta = np.linalg.solve(X.T @ X, X.T @ y)
        self.w = theta[:-1]
        self.b = theta[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """This function predicts output y based on input predictors X using the fitted model

        Args:
            X (np.ndarray): m by n predictors matrix containing m samples and n features

        Returns:
            np.ndarray: m by 1 outcome
        """
        return np.matmul(self.w, X) + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """This function fits the linear regression model using the predictors X and the outcome y.

        Args:
            X (np.ndarray): m by n predictors matrix containing m samples and n features
            y (np.ndarray): m by 1 predicted outcome matrix
            lr (float, optional): learning rate of the gradient descent. Defaults to 0.01.
            epochs (int, optional): number of traning epochs. Defaults to 1000.
        """
        self.n_samples, self.n_feats = X.shape
        self.w = np.zeros(self.n_feats)
        self.b = 0
        # gradient descent learning
        for _ in range(epochs):
            y_pred = self.predict(X)
            dw = -(2 * (X.T).dot(y - y_pred)) / self.n_samples
            db = -2 * np.sum(y - y_pred) / self.n_samples
            self.w = self.w - lr * dw
            self.b = self.b - lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.
        Arguments:
            X (np.ndarray): The input data.
        Returns:
            np.ndarray: The predicted output.
        """
        return np.matmul(self.w, X) + self.b
