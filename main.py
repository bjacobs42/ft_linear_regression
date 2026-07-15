import sys

from src.models import LinearRegression
from os import path

from src.utils import tryIntParse


def main() -> int:
    model = LinearRegression()

    if len(sys.argv) > 1:
        user_input = sys.argv[1]
    else:
        user_input = input("Enter mileage or training material: ")

    while True:
        if user_input.isdigit():
            num = tryIntParse(user_input)
            if num is not None:
                estimatedPrice = model.estimatePrice(num)
                print(f"estimatedPrice: {estimatedPrice}")
                break
        elif path.exists(user_input):
            model.train(user_input)
        else:
            print("Unknown input, try again")
        user_input = input("Enter mileage or training material: ")
    return 0


if __name__ == "__main__":
    main()
