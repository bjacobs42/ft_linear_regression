import sys

from config import USAGE
from src.models import LinearRegression

from src.utils import tryIntParse


def process_input(user_input: str, model: LinearRegression) -> int:
    """
    Processes a user command and executes the corresponding model action.

    Args:
        user_input: The command entered by the user.
        model: The linear regression model used to execute commands.

    Returns:
        0 on success, or 1 if the command or arguments are invalid.
    """

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
            print(f"Estimated price: {int(estimatedPrice)}")
        case "reset":
            model.reset()
            print("theta has been reset to 0")
        case "help":
            print(USAGE)
        case _:
            print(f'Unknown command "{args[0]}". Enter "help" for a list of commands.')

    return 0


def main() -> int:
    model = LinearRegression()

    try:
        if len(sys.argv) > 1:
            return process_input(" ".join(sys.argv[1:]), model)
        else:
            print('Type "help" for a list of commands.')
            while True:
                user_input = input("--> ")
                process_input(user_input, model)
    except EOFError:
        print()
        return 0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
