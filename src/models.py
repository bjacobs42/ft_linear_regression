from os import PathLike
from pandas.io.formats.format import math
from config import THETA_PATH
from .loading import ft_tqdm
from .utils import load, saveJson

import matplotlib.pyplot as plt
import pandas as pd
import json


class LinearRegression:
    """
    A Linear Regression model.

    Attributes:
        save_path (str | PathLike): Location of the model's saved data.
        theta0 (float): First feature in the Linear Regression model.
        theta1 (float): Second feature in the Linear Regression model.
        min0 (float): Minimum value in the dataset used to train the model.
        max0 (float): Maximum value in the dataset used to train the model.
    """

    def __init__(self, model_data=THETA_PATH) -> None:
        """
        Initializes the LinearRegression by loading its data from the path given from `model_data`.

        Args:
            model_data: Path of json file containing the model's trained data.
        """

        self.save_path = model_data
        data: dict[str, float] = {}

        try:
            with open(self.save_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            pass
        except json.JSONDecodeError as e:
            print(f"Error while loading model.json: {e}. Using thetas at 0")

        def get_number(data, key: str, default=None) -> float | None:
            value = data.get(key, default)

            try:
                if isinstance(value, (int, float)) and math.isfinite(value):
                    return value
            except Exception as e:
                print(f"Error while loading model.json: {e}. Using {key} at 0")
            return default

        self.theta0 = get_number(data, "theta0") or 0.0
        self.theta1 = get_number(data, "theta1") or 0.0
        self.min0 = get_number(data, "min0")
        self.max0 = get_number(data, "max0")

    def save(self) -> bool:
        """
        Saves the model's current parameters.

        Returns:
            True on success or False on failure.
        """

        return saveJson(
            self.save_path,
            {
                "theta0": self.theta0,
                "theta1": self.theta1,
                "min0": self.min0,
                "max0": self.max0,
            },
        )

    def reset(self) -> None:
        """
        Resets the model's data to its initial values and deletes its saved model data. (0)
        """

        self.theta0 = 0
        self.theta1 = 0
        self.min0 = None
        self.max0 = None
        self.save()

    def graph(self, file_path: str | PathLike) -> int:
        """
        Graphs the model's regression line and its errors with the dataset given by `file_path`.

        Args:
            file_path: Path of csv file containing the dataset.

        Returns:
            1 on failure and 0 on success
        """

        data = self._loadData(file_path)
        if data is None:
            return 1

        km = data.columns[0]
        price = data.columns[1]

        x = data[km]
        y = data[price]
        y_est = [self.estimatePrice(i) for i in x]

        plt.scatter(x, y, label="Data", zorder=3)
        plt.plot(x, y_est, label="Regression", color="black", zorder=2)
        plt.vlines(x, y_est, y, label="MSE", color="red", zorder=1)

        plt.xlabel("Kilometre")
        plt.ylabel("Price")

        plt.legend()
        plt.show()
        return 0

    def estimatePrice(self, kilometre: int) -> float:
        """
        Calculates the estimated price using the current model parameters.

        Args:
            kilometre: The car mileage used as the input feature.

        Returns:
            The estimated car price from the trained linear regression model.
        """

        if self.max0 is None or self.min0 is None:
            return 0

        normalized_km = (kilometre - self.min0) / (self.max0 - self.min0)
        return self._hypothesis(normalized_km)

    def train(self, filePath: str | PathLike, epoch=10000, learning_rate=0.01) -> int:
        """
        Trains the model on the dataset given and saves its training data at `self.save_path`.

        Args:
            filePath: Path of csv file containing the dataset.
            epoch: Total training iterations.
            learning_rate: The models learning rate, automatically reduces it when it overshoots.

        Returns:
            1 on failure and 0 on success.
        """
        data = self._loadData(filePath)
        if data is None:
            return 1

        data = self._normalize(data)
        if data is None:
            return 1

        prev_mse = float("inf")
        threshold = 1e-9
        print("Starting gradient descent...")
        for _ in ft_tqdm(range(epoch)):
            gradient0, gradient1 = self._computeGradient(data)

            old_theta0 = self.theta0
            old_theta1 = self.theta1

            self.theta0 -= learning_rate * gradient0
            self.theta1 -= learning_rate * gradient1

            mse = self._computeError(data)
            if mse > prev_mse:
                learning_rate *= 0.5
                print(f"\nReducing learning rate to {learning_rate}")

                self.theta0 = old_theta0
                self.theta1 = old_theta1
                continue

            if (
                abs(old_theta0 - self.theta0) < threshold
                and abs(old_theta1 - self.theta1) < threshold
            ):
                print("\nDescent finished early!")
                break

            prev_mse = mse

        return not self.save()

    def _hypothesis(self, normalized_km: float) -> float:
        """
        Calculates the estimated price using the current model parameters.

        Args:
            normalized_km: The normalized mileage value used as input.

        Returns:
            The predicted price from the linear hypothesis function.
        """

        return self.theta0 + self.theta1 * normalized_km

    def _normalize(self, data):
        """
        Normalizes data values to the range [0, 1] using min-max scaling.

        Args:
            data: a pandas DataFrame containing the dataset.

        Returns:
            The normilized DataFrame or None on failure.
        """

        col = data.columns[0]

        if self.min0 is None or self.max0 is None:
            self.min0 = int(data[col].min())
            self.max0 = int(data[col].max())

        range1 = self.max0 - self.min0
        if range1 == 0:
            return None

        data[col] = (data[col] - self.min0) / range1

        return data

    def _computeError(self, data) -> float:
        """
        Computes the Mean Squared Error (MSE) of the models hypothesis.

        Args:
            data: a pandas DataFrame containing the dataset.

        Returns:
            The Mean Squared Error between the hypothesis and actual values.
        """
        m = len(data)

        total_err = 0.0
        for mileage, price in data.itertuples(name=None, index=False):
            residual = self._hypothesis(mileage) - price
            total_err += (residual**2) / (2 * m)
        return total_err

    def _computeGradient(self, data) -> list[float]:
        """
        Computes the gradient descent updates for the linear regression parameters.

        The returned gradients represent the partial derivatives of the Mean
        Squared Error cost function with respect to theta0 and theta1.

        Args:
            data: A pandas DataFrame containing the dataset.
        Returns:
            A list containing [gradient_theta0, gradient_theta1].
        """

        m = len(data)
        gradient0 = 0.0
        gradient1 = 0.0

        for mileage, price in data.itertuples(name=None, index=False):
            residual = self._hypothesis(mileage) - price
            gradient0 += residual
            gradient1 += residual * mileage

        return [gradient0 / m, gradient1 / m]

    def _loadData(self, path: str | PathLike):
        """
        Loads a CSV file from the given path and removes duplicate rows and
        missing values.

        Args:
            path: Path to the CSV file containing the dataset.

        Returns:
            A pandas DataFrame containing the cleaned dataset.
        """

        df = load(path)
        if df is None:
            return None

        x = df.columns[0]
        y = df.columns[1]

        before = len(df)
        df[x] = pd.to_numeric(df[x], errors="coerce")
        df[y] = pd.to_numeric(df[y], errors="coerce")
        df = df.dropna()
        df = df.drop_duplicates()
        after = len(df)

        if after == 0:
            print(f"No usable data in {path}")
            return None
        if before != after:
            print(f"Dropped {before - after} amount of data due to NaN/duplicates")
        return df
