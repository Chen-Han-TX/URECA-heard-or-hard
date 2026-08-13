from pathlib import Path
import json


CONDITIONS = ["clean", "noisy", "hard"]


def load_rows(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def main():
    for condition in CONDITIONS:
        asr_path = Path(
            f"runs/asr_{condition}_labeled.jsonl"
        )

        llm_path = Path(
            f"runs/llm_{condition}_haiku.jsonl"
        )

        asr_rows = load_rows(asr_path)

        labels = {
            row["id"]: {
                "semantic_asr_failure":
                    row["semantic_asr_failure"],
                "task_critical_asr_failure":
                    row["task_critical_asr_failure"],
                "semantic_failure_reason":
                    row["semantic_failure_reason"],
            }
            for row in asr_rows
        }

        llm_rows = load_rows(llm_path)

        for row in llm_rows:
            label = labels[row["id"]]

            row.update(label)

        with llm_path.open("w", encoding="utf-8") as f:
            for row in llm_rows:
                f.write(json.dumps(row) + "\n")

        print(
            f"{condition}: refreshed "
            f"{len(llm_rows)} LLM rows"
        )


if __name__ == "__main__":
    main()