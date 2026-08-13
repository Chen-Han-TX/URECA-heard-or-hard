import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from run_llm import build_prompt


def main():
    prompt = build_prompt(
        "What is the confirmation code K7M42?"
    )

    assert "Use ONLY the information" in prompt
    assert "Do not look up external information" in prompt
    assert "return that value directly" in prompt

    print("All LLM prompt tests passed.")


if __name__ == "__main__":
    main()