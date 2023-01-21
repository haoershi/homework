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
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_
        """
        self.n_samples, self.n_feats = X.shape
        X = np.c_[X, np.ones(self.n_samples)]
        theta = np.linalg.solve(X.T @ X, X.T @ y)
        self.w = theta[:-1]
        self.b = theta[-1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        assert (
            X.shape[1] == self.n_feats,
            "The input number of features inconsistent with model.",
        )
        return np.matmul(self.w, X) + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """_summary_

        Args:
            X (np.ndarray): _description_
            y (np.ndarray): _description_
            lr (float, optional): _description_. Defaults to 0.01.
            epochs (int, optional): _description_. Defaults to 1000.
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
