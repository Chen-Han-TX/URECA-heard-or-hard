from pathlib import Path
import json


CONDITIONS = ["clean", "noisy", "hard"]


def load_rows(condition):
    path = Path(
        f"runs/llm_{condition}_haiku_scored.jsonl"
    )

    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def fmt(value):
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def analyze_condition(condition):
    rows = load_rows(condition)

    failures = [
        row for row in rows
        if row["task_critical_asr_failure"]
    ]

    non_failures = [
        row for row in rows
        if not row["task_critical_asr_failure"]
    ]

    correct = sum(
        row["correct"]
        for row in rows
    )

    perception_failures = [
        row for row in rows
        if (
            row["task_critical_asr_failure"]
            and not row["correct"]
        )
    ]

    return {
        "condition": condition,
        "samples": len(rows),
        "accuracy": correct / len(rows),
        "task_critical_failures": len(failures),
        "perception_failures": len(perception_failures),

        "asr_logprob_failure": mean([
            row["avg_logprob"]
            for row in failures
        ]),

        "asr_logprob_normal": mean([
            row["avg_logprob"]
            for row in non_failures
        ]),

        "llm_conf_failure": mean([
            row["self_confidence"]
            for row in perception_failures
        ]),

        "llm_conf_correct": mean([
            row["self_confidence"]
            for row in rows
            if row["correct"]
        ]),

        "mean_ttft_ms": mean([
            row["llm_ttft_ms"]
            for row in rows
        ]),

        "mean_total_ms": mean([
            row["llm_total_ms"]
            for row in rows
        ]),
    }


def main():
    results = [
        analyze_condition(condition)
        for condition in CONDITIONS
    ]

    print()
    print("CONDITION SUMMARY")
    print("=" * 90)

    for row in results:
        print()
        print(row["condition"].upper())

        print(
            f"  Samples                  : "
            f"{row['samples']}"
        )

        print(
            f"  Accuracy                 : "
            f"{row['accuracy']:.1%}"
        )

        print(
            f"  Task-critical ASR errors : "
            f"{row['task_critical_failures']}"
        )

        print(
            f"  Perception failures      : "
            f"{row['perception_failures']}"
        )

        print(
            f"  ASR logprob | failure    : "
            f"{fmt(row['asr_logprob_failure'])}"
        )

        print(
            f"  ASR logprob | normal     : "
            f"{fmt(row['asr_logprob_normal'])}"
        )

        print(
            f"  LLM conf | perception    : "
            f"{fmt(row['llm_conf_failure'])}"
        )

        print(
            f"  LLM conf | correct       : "
            f"{fmt(row['llm_conf_correct'])}"
        )

        print(
            f"  Mean TTFT                : "
            f"{fmt(row['mean_ttft_ms'])} ms"
        )

        print(
            f"  Mean total latency       : "
            f"{fmt(row['mean_total_ms'])} ms"
        )


if __name__ == "__main__":
    main()