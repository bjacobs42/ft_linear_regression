from load_csv import load
import sys


def main() -> int:

    if (len(sys.argv) < 2):
        print("This is how you do it~")  # print user manual
        return 0

    file_path: str = sys.argv[1]
    assert isinstance(file_path, str)

    df = load(file_path)
    if (df is None):
        return 0
    print(df)
    # train using gradient descent
    # save theta0 and theta1 to a file

    return (0)


if __name__ == "__main__":
    main()
