import os, sys


def function_int(x: int, y: int) -> int:
    return x + y


def function_2() -> str:
    return "function_2"


def main() -> None:
    print(f"Result in INT: {function_int(1, 2)}")


if __name__ == "__main__":
    main()


# support int, suport int + float.
