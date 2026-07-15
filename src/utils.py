import pandas as pd


def saveData(path: str, columns: list[str], data, silent=False) -> bool:
    try:
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(path, index=False)
    except (ValueError, TypeError) as e:
        if not silent:
            print(f"saveData: Invalid data: {e}")
        return False
    except (FileNotFoundError, PermissionError, IsADirectoryError, OSError) as e:
        if not silent:
            print(f"saveData: Couldn't save file: {e}")
        return False
    return True


def load(path: str, silent=False) -> pd.DataFrame | None:
    """
    Uses pandas to load a file specified by `path`.
    Returns a pandas DataFrame or None on error.
    """

    try:
        df = pd.read_csv(path, index_col=0)
    except (FileNotFoundError, UnicodeDecodeError, pd.errors.ParserError) as e:
        if not silent:
            print(f"Error: {e}")
        return None
    except pd.errors.EmptyDataError:
        if not silent:
            print("Error: empty file")
        return None

    # print(f"loading dataset of dimensions {df.shape}")
    return df


def tryIntParse(userInput: str) -> int:
    try:
        return int(userInput)
    except ValueError:
        return -1


def parseInput(argv: list) -> int:
    userInput: str
    value: int

    if len(argv) > 1:
        userInput = argv[1]
    else:
        userInput = input("Enter mileage: ")

    value = tryIntParse(userInput)
    while value < 0:
        print("Invalid mileage, please try again")
        userInput = input("Re-enter mileage: ")
        value = tryIntParse(userInput)

    return value
