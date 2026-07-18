import json
from os import PathLike
import pandas as pd

from typing import Any


def saveJson(path: str | PathLike, data: dict[str, Any]) -> bool:
    """
    Uses json to save a json file specified by `path`.

    Returns True on success and False on failure.
    """

    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
    except (PermissionError, IsADirectoryError, OSError) as e:
        print(f"saveJson: Couldn't save file: {e}")
        return False
    return True


def saveCSV(path: str | PathLike, columns: list[str], data, silent=False) -> bool:
    """
    Uses pandas to save a csv file specified by `path`.
    The csv file will contain the headers specified by `columns` and its rows specified by `data`.
    Errors can be silenced by setting `silent` to True.

    Returns True on success and False on failure.
    """

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


def load(path: str | PathLike, silent=False) -> pd.DataFrame | None:
    """
    Uses pandas to load a csv file specified by `path`.
    Errors can be silenced by setting `silent` to True.
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
    """
    Tries to cast a str to int with try except protection.
    Errors can be silenced by setting `silent` to True.

    Returns an int or None on failure.
    """

    try:
        return int(userInput)
    except ValueError as e:
        if not silent:
            print(f"ValueError: {e}")
        return None
