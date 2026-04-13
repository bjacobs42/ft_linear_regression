import pandas as pd
from load_csv import load
from estimate_price import estimatePrice
import sys


def mean_squared_error(predicted_y: float, real_y: float) -> float:
    # basically this should be:
    # 1/n * (sum of (predicted_yi - real_yi)^2)
    return (real_y - predicted_y) ** 2


def linearRegression(data: pd.DataFrame):
    # learn partial derivatives and start again!
    return [theta0, theta1]


def main() -> int:

    if len(sys.argv) < 2:
        print("This is how you do it~")  # print user manual
        return 0

    file_path: str = sys.argv[1]
    assert isinstance(file_path, str)

    df = load(file_path)
    if df is None:
        return 0
    print(df)
    theta: list = linearRegression()
    # save theta0 and theta1 to a file

    return 0


if __name__ == "__main__":
    main()
