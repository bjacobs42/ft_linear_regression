import pandas as pd


def load(path: str) -> pd.DataFrame:
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

    print(f"loading dataset of dimensions {df.shape}")
    return df
