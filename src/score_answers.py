from pathlib import Path
import json
import re
import sys

from normalize import normalize_text


def normalize_answer(text: str) -> str:
    text = text.lower().strip()

    # Remove common punctuation/formatting differences.
    text = text.replace("$", "")

    # Normalize explicit clock forms.
    # 4.45 -> 4:45
    if re.fullmatch(r"\d{1,2}\.\d{2}", text):
        hour, minute = text.split(".")
        text = f"{int(hour)}:{minute}"

    # 4:00 PM -> 4
    match = re.fullmatch(
        r"(\d{1,2}):00\s*(am|pm)?",
        text,
    )

    if match:
        text = str(int(match.group(1)))

    # Remove units that should not affect correctness.
    text = re.sub(
        r"\b(items?|dollars?|dollar)\b",
        "",
        text,
    )

    text = re.sub(r"\s+", " ", text).strip()

    # Reuse general text/number normalization.
    text = normalize_text(text)

    return text

def is_correct(prediction: str, gold: str) -> bool:
    pred = normalize_answer(prediction)
    target = normalize_answer(gold)

    return pred == target


def main():
    if len(sys.argv) != 2:
        sys.exit(
            "Usage: python src/score_answers.py "
            "<clean|noisy|hard>"
        )

    condition = sys.argv[1]

    input_path = Path(
        f"runs/llm_{condition}_haiku.jsonl"
    )

    output_path = Path(
        f"runs/llm_{condition}_haiku_scored.jsonl"
    )

    rows = [
        json.loads(line)
        for line in input_path.read_text().splitlines()
        if line.strip()
    ]

    correct_count = 0

    for row in rows:
        prediction = row["answer"]
        gold = row["gold"]

        normalized_prediction = normalize_answer(
            prediction
        )

        normalized_gold = normalize_answer(
            gold
        )

        correct = (
            normalized_prediction
            == normalized_gold
        )

        row["normalized_answer"] = (
            normalized_prediction
        )

        row["normalized_gold"] = normalized_gold

        row["correct"] = correct

        correct_count += int(correct)

        print(
            f"{row['id']}: "
            f"pred={prediction!r} "
            f"gold={gold!r} "
            f"-> {correct}"
        )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    accuracy = correct_count / len(rows)

    print()
    print(
        f"Accuracy: "
        f"{correct_count}/{len(rows)} "
        f"= {accuracy:.1%}"
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()