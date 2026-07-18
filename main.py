import sys

from src.models import LinearRegression
from os import path

from src.utils import tryIntParse


def process_input(user_input: str, model: LinearRegression):
    if user_input.isdigit():
        num = tryIntParse(user_input)
        if num is None:
            return 1
        estimatedPrice = model.estimatePrice(num)
        print(f"estimatedPrice: {estimatedPrice}")
    elif path.exists(user_input):
        model.train(user_input)
    else:
        print("Unknown input, try again")
        return 1
    return 0


def main() -> int:
    model = LinearRegression()

    if len(sys.argv) > 1:
        return process_input(sys.argv[1], model)
    else:
        while True:
            user_input = input("Enter mileage or training material: ")
            process_input(user_input, model)
    return 0


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print()
