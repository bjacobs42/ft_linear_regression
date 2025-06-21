import sys


def tryIntParse(userInput: str) -> int:
    try:
        return int(userInput)
    except ValueError:
        return (-1)


def parseInput() -> int:
    userInput: str
    value: int

    if (len(sys.argv) > 1):
        userInput = sys.argv[1]
    else:
        userInput = input("Enter mileage: ")

    value = tryIntParse(userInput)
    while value < 0:
        print("Invalid mileage, please try again")
        userInput = input("Re-enter mileage: ")
        value = tryIntParse(userInput)

    return (value)


def main() -> int:
    mileage: int = parseInput()
    # get theta1 and 0
    # print theta0 + theta1 * mileage

    print(mileage)
    return (0)


if __name__ == "__main__":
    main()
