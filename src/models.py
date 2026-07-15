from .utils import load, saveData
import math

THETA_PATH = "../data/theta.csv"


class LinearRegression:
    def __init__(self, learning_rate=0.000001) -> None:
        self.theta0, self.theta1 = self._loadTheta()
        self.learning_rate = learning_rate

    def graph(self) -> None:
        pass

    def estimatePrice(self, mileage: int) -> float:
        return self.theta0 + self.theta1 * mileage

    def train(self, filePath: str) -> None:
        data = load(filePath)
        if data is None:
            return

        # format data (Remove duplicates, change from str to int, check format [num, num])
        data.iloc[:, 0] = data.iloc[:, 0].astype(int)
        data.iloc[:, 1] = data.iloc[:, 1].astype(int)

        prev_gradient1 = None
        prev_gradient0 = None

        m = len(data)
        epsilon = 1e-6
        while True:
            gradient0 = 0.0
            gradient1 = 0.0

            for mileage, price in data.itertuples(name=None, index=False):
                residual = self.estimatePrice(mileage) - price
                if math.isinf(residual):
                    break
                print(residual)
                gradient0 += residual
                gradient1 += residual * mileage

            if prev_gradient0 is not None and prev_gradient1 is not None:
                if (
                    abs(prev_gradient0 - gradient0) <= epsilon
                    and abs(prev_gradient1 - gradient1) <= epsilon
                ):
                    break

            self.theta0 += self.learning_rate * (gradient0) / m
            self.theta1 += self.learning_rate * (gradient1) / m

            prev_gradient0 = gradient0
            prev_gradient1 = gradient1

        saveData(THETA_PATH, ["theta0", "theta1"], [[self.theta0, self.theta1]])

    def _loadTheta(self) -> list[float]:
        theta = load(THETA_PATH, silent=True)
        if theta is None or len(theta) != 2:
            return [0.0, 0.0]
        return theta.iloc[0].astype(float).tolist()
