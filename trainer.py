from load_csv import load
import sys


def main() -> int:

    if (len(sys.argv) < 2):
        print("This is how you do it~")  # print user manual
        return 0

    file_path: str = sys.argv[1]
    assert isinstance(file_path, str)

    df = load(file_path)
    print(df)

    return (0)


if __name__ == "__main__":
    main()
