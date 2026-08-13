from pathlib import Path
import json


CONDITIONS = ["clean", "noisy", "hard"]


def load_results(condition):
    path = Path(f"runs/asr_{condition}.jsonl")

    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def main():
    for condition in CONDITIONS:
        rows = load_results(condition)

        print()
        print("=" * 70)
        print(condition.upper())
        print("=" * 70)

        errors = [
            row
            for row in rows
            if row["normalized_wer"] > 0
        ]

        if not errors:
            print("No normalized transcription errors.")
            continue

        for row in errors:
            print()
            print(f"ID       : {row['id']}")
            print(f"Category : {row['category']}")
            print(f"WER      : {row['normalized_wer']:.3f}")
            print(f"Logprob  : {row['avg_logprob']:.3f}")
            print(f"REF      : {row['normalized_reference']}")
            print(f"PRED     : {row['normalized_transcript']}")


if __name__ == "__main__":
    main()