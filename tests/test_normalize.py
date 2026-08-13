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
    
    check(
    "What room number did they give you? 426.",
    "what room number did they give you 426",
    )

    check(
        "If a train leaves at two and arrives three hours later",
        "if a train leaves at 2 and arrives 3 hours later",
    )

    check(
        "four hundred and twenty six",
        "426",
    )

    check(
        "27 dollars and 50 cents",
        "27.50",
    )

    check(
        "27 dollars",
        "27",
    )

    check(
        "How many dollars should I transfer, 27 dollars and 50 cents?",
        "how many dollars should i transfer 27.50",
    )
    
    check(
    "What time is the meeting, four fifteen?",
    "what time is the meeting 4:15",
)

    check(
        "What time is the meeting, four fifty?",
        "what time is the meeting 4:50",
    )

    check(
        "If the meeting starts at four fifteen and lasts thirty minutes",
        "if the meeting starts at 4:15 and lasts 30 minutes",
    )

    check(
        "March thirtieth",
        "march 30",
    )
    
    check(
    "What time is the meeting, 4.15?",
    "what time is the meeting 4:15",
)

    check(
        "If the meeting starts at 4.50 and lasts 30 minutes",
        "if the meeting starts at 4:50 and lasts 30 minutes",
    )
    
    
    check(
    "What time is the meeting 450",
    "what time is the meeting 4:50",
    )

    check(
        "The meeting starts at 415",
        "the meeting starts at 4:15",
    )
    
    check(
    "The price is 450",
    "the price is 450",
)
    print("All normalization tests passed.")


if __name__ == "__main__":
    main()