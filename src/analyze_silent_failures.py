from pathlib import Path
import json


CONDITIONS = ["clean", "noisy", "hard"]
HIGH_CONFIDENCE_THRESHOLD = 80


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
    all_failures = []

    for condition in CONDITIONS:
        rows = load_rows(condition)

        for row in rows:
            if (
                row["task_critical_asr_failure"]
                and not row["correct"]
            ):
                all_failures.append(row)

    silent = [
        row for row in all_failures
        if row["self_confidence"]
        >= HIGH_CONFIDENCE_THRESHOLD
    ]

    detected = [
        row for row in all_failures
        if row["self_confidence"]
        < HIGH_CONFIDENCE_THRESHOLD
    ]

    print("PERCEPTION FAILURE ANALYSIS")
    print("=" * 70)

    print(
        f"Total perception failures : "
        f"{len(all_failures)}"
    )

    print(
        f"High-confidence failures  : "
        f"{len(silent)}"
    )

    print(
        f"Low-confidence failures   : "
        f"{len(detected)}"
    )

    if all_failures:
        print(
            f"Silent failure rate       : "
            f"{len(silent) / len(all_failures):.1%}"
        )

    print()
    print("HIGH-CONFIDENCE PERCEPTION FAILURES")
    print("=" * 70)

    for row in silent:
        print()
        print(
            f"{row['condition']}/{row['id']}"
        )
        print(
            f"Transcript : {row['transcript']}"
        )
        print(
            f"Answer     : {row['answer']}"
        )
        print(
            f"Gold       : {row['gold']}"
        )
        print(
            f"Confidence : "
            f"{row['self_confidence']}"
        )
        print(
            f"ASR logprob: "
            f"{row['avg_logprob']:.3f}"
        )


if __name__ == "__main__":
    main()