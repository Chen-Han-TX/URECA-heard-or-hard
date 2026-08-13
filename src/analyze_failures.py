from pathlib import Path
import json


CONDITIONS = ["clean", "noisy", "hard"]


def classify(row):
    asr_failed = row["semantic_asr_failure"]
    correct = row["correct"]

    if not asr_failed and correct:
        return "normal"

    if not asr_failed and not correct:
        return "reasoning_failure"

    if asr_failed and correct:
        return "llm_rescue"

    return "perception_failure"


def load_rows(condition):
    path = Path(
        f"runs/llm_{condition}_haiku_scored.jsonl"
    )

    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def main():
    for condition in CONDITIONS:
        rows = load_rows(condition)

        counts = {
            "normal": 0,
            "reasoning_failure": 0,
            "llm_rescue": 0,
            "perception_failure": 0,
        }

        print()
        print("=" * 70)
        print(condition.upper())
        print("=" * 70)

        for row in rows:
            label = classify(row)
            counts[label] += 1

            if label != "normal":
                print()
                print(f"ID         : {row['id']}")
                print(f"Type       : {label}")
                print(f"Transcript : {row['transcript']}")
                print(f"Answer     : {row['answer']}")
                print(f"Gold       : {row['gold']}")
                print(
                    f"LLM conf   : "
                    f"{row['self_confidence']}"
                )
                print(
                    f"ASR logprob: "
                    f"{row['avg_logprob']:.3f}"
                )

        print()
        print("Summary")
        print(f"  normal             : {counts['normal']}")
        print(
            f"  reasoning failure  : "
            f"{counts['reasoning_failure']}"
        )
        print(
            f"  LLM rescue         : "
            f"{counts['llm_rescue']}"
        )
        print(
            f"  perception failure : "
            f"{counts['perception_failure']}"
        )


if __name__ == "__main__":
    main()