import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from run_llm import parse_response


def main():
    plain = """
{
  "answer": "14.70",
  "confidence": 95
}
"""

    fenced = """```json
{
  "answer": "$14.70",
  "confidence": 99
}
```"""

    result1 = parse_response(plain)
    result2 = parse_response(fenced)

    assert result1["answer"] == "14.70"
    assert result1["confidence"] == 95

    assert result2["answer"] == "$14.70"
    assert result2["confidence"] == 99

    print("All LLM parser tests passed.")


if __name__ == "__main__":
    main()