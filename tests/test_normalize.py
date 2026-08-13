import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from normalize import normalize_text


def check(source, expected):
    actual = normalize_text(source)

    print(f"INPUT   : {source}")
    print(f"EXPECTED: {expected}")
    print(f"ACTUAL  : {actual}")
    print()

    assert actual == expected


def main():
    check(
        "three",
        "3",
    )

    check(
        "seventeenth",
        "17",
    )

    check(
        "17th",
        "17",
    )

    check(
        "four hundred and twenty six",
        "426",
    )

    check(
        "four dollars",
        "4",
    )

    check(
        "twenty seven dollars and fifty cents",
        "27.50",
    )

    check(
        "four dollars ninety",
        "4.90",
    )

    print("All normalization tests passed.")


if __name__ == "__main__":
    main()