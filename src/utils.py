import pandas as pd


def load(path: str) -> pd.DataFrame | None:
    """
    Uses pandas to load a file specified by `path`.
    Returns a pandas DataFrame or None on error.
    """

    try:
        df = pd.read_csv(path, index_col=0)
    except (FileNotFoundError, UnicodeDecodeError, pd.errors.ParserError) as e:
        print(f"Error: {e}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: empty file")
        return None

    # print(f"loading dataset of dimensions {df.shape}")
    return df


def tryIntParse(userInput: str) -> int:
    try:
        return int(userInput)
    except ValueError:
        return -1


def parseInput() -> int:
    userInput: str
    value: int

    if len(sys.argv) > 1:
        userInput = sys.argv[1]
    else:
        userInput = input("Enter mileage: ")

    value = tryIntParse(userInput)
    while value < 0:
        print("Invalid mileage, please try again")
        userInput = input("Re-enter mileage: ")
        value = tryIntParse(userInput)

    return value


def getTheta() -> list:
    theta = ft_load("./theta.csv")
    if theta is None:
        return [0, 0]
    # double check format here!! (Must be a [int, int])
    return theta
