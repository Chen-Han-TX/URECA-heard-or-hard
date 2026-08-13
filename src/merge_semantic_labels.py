from pathlib import Path
import json


LABEL_PATH = Path("data/asr_semantic_labels.jsonl")


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

    for condition in ["clean", "noisy", "hard"]:
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

            if condition == "clean":
                row["semantic_asr_failure"] = False
                row["semantic_failure_reason"] = None

            elif label:
                row["semantic_asr_failure"] = label[
                    "semantic_asr_failure"
                ]
                row["semantic_failure_reason"] = label[
                    "reason"
                ]

            else:
                # No normalized transcription error means
                # no ASR semantic failure in this pilot.
                if row["normalized_wer"] == 0:
                    row["semantic_asr_failure"] = False
                    row["semantic_failure_reason"] = None
                else:
                    raise ValueError(
                        f"Missing manual label for "
                        f"{condition}/{row['id']}"
                    )

            rows.append(row)

        with output_path.open("w") as f:
            for row in rows:
                f.write(
                    json.dumps(row) + "\n"
                )

        failures = sum(
            row["semantic_asr_failure"]
            for row in rows
        )

        print(
            f"{condition}: "
            f"{failures}/{len(rows)} "
            f"semantic failures"
        )


if __name__ == "__main__":
    main()