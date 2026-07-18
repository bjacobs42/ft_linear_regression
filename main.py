import sys

from config import USAGE
from src.models import LinearRegression

from src.utils import tryIntParse


def process_input(user_input: str, model: LinearRegression) -> int:
    args = user_input.split(" ")

    if len(args) == 0:
        return 0

    match args[0].lower():
        case "graph":
            if len(args) != 2:
                print("Usage: graph [csv_file]")
                return 1
            return model.graph(args[1])
        case "train":
            if len(args) != 2:
                print("Usage: train [csv_file]")
                return 1
            return model.train(args[1])
        case "estimate":
            if len(args) != 2:
                print("Usage: estimate [km]")
                return 1
            num = tryIntParse(args[1])
            if num is None:
                return 1
            estimatedPrice = model.estimatePrice(num)
            print(f"Estimated price: {estimatedPrice}")
        case "reset":
            model.reset()
            print("theta has been reset to 0")
        case "help":
            print(USAGE)
        case _:
            print(f'Unknown command "{args[0]}". Type "help" for a list of commands.')

    return 0


def main() -> int:
    model = LinearRegression()

    if len(sys.argv) > 1:
        return process_input(sys.argv[1], model)
    else:
        print('Type "help" for a list of commands.')
        while True:
            user_input = input(">>> ")
            process_input(user_input, model)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print()
