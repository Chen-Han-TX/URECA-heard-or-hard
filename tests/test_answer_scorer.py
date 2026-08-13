import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from score_answers import is_correct


def main():
    assert is_correct("$14.70", "14.70")
    assert is_correct("$12", "12")
    assert is_correct("24 items", "24")
    assert is_correct("5", "five")
    assert is_correct("October 17th", "October seventeenth")
    assert is_correct("Thursday", "Thursday")
    assert is_correct("K7M42", "K7M42")

    assert not is_correct("14.20", "14.70")
    assert not is_correct("Tuesday", "Thursday")

    print("All answer scorer tests passed.")


if __name__ == "__main__":
    main()