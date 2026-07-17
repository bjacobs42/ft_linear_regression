from config import THETA_PATH
from .loading import ft_tqdm
from .utils import load, saveJson
import pandas as pd
import json


class LinearRegression:
    def __init__(self) -> None:
        data: dict[str, float] = {}

        try:
            with open(THETA_PATH, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            pass
        except json.JSONDecodeError as e:
            print(f"Error while loading model.json: {e}. Using thetas at 0")

        self.theta0 = data.get("theta0") or 0.0
        self.theta1 = data.get("theta1") or 0.0
        self.min0 = data.get("min0")
        self.max0 = data.get("max0")

    def graph(self) -> None:
        pass

    def estimatePrice(self, mileage: int) -> float:
        if self.max0 is None or self.min0 is None:
            return 0

        normalized_mileage = (mileage - self.min0) / (self.max0 - self.min0)
        return self.theta0 + self.theta1 * normalized_mileage

    def train(self, filePath: str, epoch=10000, learning_rate=0.01) -> None:
        data = self._loadData(filePath)
        if data is None:
            return

        data = self._normalize(data)

        threshold = 1e-6
        print("Starting gradient descent...")
        for _ in ft_tqdm(range(epoch)):
            gradient0, gradient1 = self._computeGradient(data)

            old_theta0 = self.theta0
            old_theta1 = self.theta1

            self.theta0 -= learning_rate * gradient0
            self.theta1 -= learning_rate * gradient1
            # print(
            #    f"gradients: {gradient0}, {gradient1}",
            #    f"thetas: {self.theta0}, {self.theta1}",
            #    f"epoch: {_}",
            # )

            if (
                abs(old_theta0 - self.theta0) < threshold
                and abs(old_theta1 - self.theta1) < threshold
            ):
                print("\nDescent finished early!")
                break

        saveJson(
            THETA_PATH,
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

    def _computeGradient(self, data) -> list[float]:
        m = len(data)
        gradient0 = 0.0
        gradient1 = 0.0

        for mileage, price in data.itertuples(name=None, index=False):
            residual = self._hypothesis(mileage) - price
            gradient0 += residual
            gradient1 += residual * mileage

        return [gradient0 / m, gradient1 / m]

    def _loadData(self, path: str):
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
