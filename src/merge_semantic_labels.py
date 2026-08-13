from pathlib import Path
import json


LABEL_PATH = Path("data/asr_semantic_labels.jsonl")
CONDITIONS = ["clean", "noisy", "hard"]


def load_labels():
    labels = {}

    for line in LABEL_PATH.read_text().splitlines():
        if not line.strip():
            continue

        row = json.loads(line)

        key = (
            row["condition"],
            row["id"],
        )

        labels[key] = row

    return labels


def main():
    labels = load_labels()

    for condition in CONDITIONS:
        input_path = Path(
            f"runs/asr_{condition}.jsonl"
        )

        output_path = Path(
            f"runs/asr_{condition}_labeled.jsonl"
        )

        rows = []

        for line in input_path.read_text().splitlines():
            if not line.strip():
                continue

            row = json.loads(line)

            key = (
                condition,
                row["id"],
            )

            label = labels.get(key)

            # Explicit manual annotation always takes priority.
            if label:
                row["semantic_asr_failure"] = label[
                    "semantic_asr_failure"
                ]

                row["task_critical_asr_failure"] = label[
                    "task_critical_asr_failure"
                ]

                row["semantic_failure_reason"] = label[
                    "reason"
                ]

            # No normalized transcription difference:
            # treat as no ASR failure.
            elif row["normalized_wer"] == 0:
                row["semantic_asr_failure"] = False
                row["task_critical_asr_failure"] = False
                row["semantic_failure_reason"] = None

            # Any WER > 0 sample must be manually reviewed.
            else:
                raise ValueError(
                    f"Missing manual label for "
                    f"{condition}/{row['id']} "
                    f"(normalized_wer="
                    f"{row['normalized_wer']:.3f})"
                )

            rows.append(row)

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        semantic_failures = sum(
            row["semantic_asr_failure"]
            for row in rows
        )

        task_critical_failures = sum(
            row["task_critical_asr_failure"]
            for row in rows
        )

        print(
            f"{condition}: "
            f"{semantic_failures}/{len(rows)} "
            f"semantic failures, "
            f"{task_critical_failures}/{len(rows)} "
            f"task-critical failures"
        )


if __name__ == "__main__":
    main()