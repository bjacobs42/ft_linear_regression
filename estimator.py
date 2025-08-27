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


def getTheta() -> list:
    theta = ft_load("./theta.csv")
    if theta is None:
        return [0, 0]
    return theta


def main() -> int:
    mileage: int = parseInput()
    theta: list = getTheta()
    estimatedPrice: int = theta[0] + theta[1] * mileage

    print("Estimated price for mileage " + str(mileage) + ": " str(estimatedPrice))
    return (0)


if __name__ == "__main__":
    main()
