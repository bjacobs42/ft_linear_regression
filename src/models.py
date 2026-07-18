from os import PathLike

from pandas.io.formats.format import math
from config import THETA_PATH
from .loading import ft_tqdm
from .utils import load, saveJson
import pandas as pd
import json


class LinearRegression:
    def __init__(self, model_data=THETA_PATH) -> None:
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

    def reset(self) -> None:
        self.theta0 = 0
        self.theta1 = 0
        self.min0 = None
        self.max0 = None

    def graph(self, filePath: str | PathLike) -> int:
        data = self._loadData(filePath)
        if data is None:
            return 1
        return 0

    def estimatePrice(self, mileage: int) -> float:
        if self.max0 is None or self.min0 is None:
            return 0

        normalized_mileage = (mileage - self.min0) / (self.max0 - self.min0)
        return self.theta0 + self.theta1 * normalized_mileage

    def train(self, filePath: str | PathLike, epoch=10000, learning_rate=0.01) -> int:
        data = self._loadData(filePath)
        if data is None:
            return 1

        data = self._normalize(data)

        prev_err = float("inf")
        threshold = 1e-9
        print("Starting gradient descent...")
        for _ in ft_tqdm(range(epoch)):
            gradient0, gradient1 = self._computeGradient(data)

            old_theta0 = self.theta0
            old_theta1 = self.theta1

            self.theta0 -= learning_rate * gradient0
            self.theta1 -= learning_rate * gradient1

            err = self._computeError(data)
            if err > prev_err:
                learning_rate *= 0.5
                print(f"Reducing learning rate to {learning_rate}")

                self.theta0 = old_theta0
                self.theta1 = old_theta1
                continue

            if abs(prev_err - err) < threshold:
                print("\nDescent finished early!")
                break

            prev_err = err

        return not saveJson(
            self.save_path,
            {
                "theta0": self.theta0,
                "theta1": self.theta1,
                "min0": float(self.min0 or 0.0),
                "max0": float(self.max0 or 0.0),
            },
        )

    def _hypothesis(self, normalized_mileage: float) -> float:
        return self.theta0 + self.theta1 * normalized_mileage

    def _normalize(self, data):
        col = data.columns[0]

        if self.min0 is None or self.max0 is None:
            self.min0 = data[col].min()
            self.max0 = data[col].max()

        range1 = self.max0 - self.min0
        if range1 == 0:
            raise ValueError("feature1 values are all identical")

        data[col] = (data[col] - self.min0) / range1

        return data

    def _computeError(self, data) -> float:
        m = len(data)

        total_err = 0.0
        for mileage, price in data.itertuples(name=None, index=False):
            residual = self._hypothesis(mileage) - price
            total_err += (residual**2) / (2 * m)
        return total_err

    def _computeGradient(self, data) -> list[float]:
        m = len(data)
        gradient0 = 0.0
        gradient1 = 0.0

        for mileage, price in data.itertuples(name=None, index=False):
            residual = self._hypothesis(mileage) - price
            gradient0 += residual
            gradient1 += residual * mileage

        return [gradient0 / m, gradient1 / m]

    def _loadData(self, path: str | PathLike):
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
