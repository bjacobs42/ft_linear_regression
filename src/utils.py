import json
import pandas as pd

from typing import Any


def saveJson(path: str, data: dict[str, Any]) -> bool:
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
    except (PermissionError, IsADirectoryError, OSError) as e:
        print(f"saveJson: Couldn't save file: {e}")
        return False
    return True


def saveCSV(path: str, columns: list[str], data, silent=False) -> bool:
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
        df = pd.read_csv(path)
    except (
        IsADirectoryError,
        FileNotFoundError,
        UnicodeDecodeError,
        PermissionError,
        pd.errors.ParserError,
    ) as e:
        if not silent:
            print(f"Error: {e}")
        return None
    except pd.errors.EmptyDataError:
        if not silent:
            print("Error: empty file")
        return None

    # print(f"loading dataset of dimensions {df.shape}")
    return df


def tryIntParse(userInput: str, silent=False) -> int | None:
    try:
        return int(userInput)
    except ValueError as e:
        if not silent:
            print(f"ValueError: {e}")
        return None
