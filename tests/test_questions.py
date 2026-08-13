from pathlib import Path
import json


QUESTION_PATH = Path("data/questions.jsonl")


def main():
    rows = [
        json.loads(line)
        for line in QUESTION_PATH.read_text().splitlines()
        if line.strip()
    ]

    assert len(rows) == 48

    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))

    entity = [
        row for row in rows
        if row["category"] == "entity"
    ]

    reason = [
        row for row in rows
        if row["category"] == "reason"
    ]

    assert len(entity) == 24
    assert len(reason) == 24

    for row in rows:
        assert row["id"]
        assert row["text"]
        assert row["gold"]

    print("Questions:", len(rows))
    print("Entity:", len(entity))
    print("Reason:", len(reason))
    print("All dataset tests passed.")


if __name__ == "__main__":
    main()