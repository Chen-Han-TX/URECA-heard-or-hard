from pathlib import Path
import json

import matplotlib.pyplot as plt


CONDITIONS = ["clean", "noisy", "hard"]

DISPLAY_NAMES = {
    "clean": "Clean",
    "noisy": "5 dB Noise",
    "hard": "Speed + Bandwidth",
}

OUTPUT_DIR = Path("plots")


def load_rows(condition):
    path = Path(
        f"runs/llm_{condition}_haiku_scored.jsonl"
    )

    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def collect_stats():
    stats = {}

    for condition in CONDITIONS:
        rows = load_rows(condition)

        correct = sum(
            row["correct"]
            for row in rows
        )

        perception_failures = [
            row
            for row in rows
            if (
                row["task_critical_asr_failure"]
                and not row["correct"]
            )
        ]

        stats[condition] = {
            "samples": len(rows),
            "accuracy": correct / len(rows),
            "perception_failures": len(
                perception_failures
            ),
        }

    return stats


def plot_accuracy(stats):
    labels = [
        DISPLAY_NAMES[c]
        for c in CONDITIONS
    ]

    values = [
        stats[c]["accuracy"] * 100
        for c in CONDITIONS
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        labels,
        values,
    )

    ax.set_ylabel("Final Answer Accuracy (%)")
    ax.set_title(
        "End-to-End Accuracy by Audio Condition"
    )

    ax.set_ylim(0, 105)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    output = OUTPUT_DIR / "accuracy_by_condition.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)

    print(f"Saved: {output}")


def plot_perception_failures(stats):
    labels = [
        DISPLAY_NAMES[c]
        for c in CONDITIONS
    ]

    values = [
        stats[c]["perception_failures"]
        for c in CONDITIONS
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        labels,
        values,
    )

    ax.set_ylabel(
        "Number of Perception Failures"
    )

    ax.set_title(
        "Task-Critical ASR Failures "
        "Propagating to Final Answers"
    )

    ax.set_ylim(
        0,
        max(values) + 2,
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.2,
            str(value),
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    output = (
        OUTPUT_DIR
        / "perception_failures_by_condition.png"
    )

    fig.savefig(output, dpi=200)
    plt.close(fig)

    print(f"Saved: {output}")


def plot_failure_confidence():
    failures = []

    for condition in CONDITIONS:
        rows = load_rows(condition)

        for row in rows:
            if (
                row["task_critical_asr_failure"]
                and not row["correct"]
            ):
                failures.append(row)

    high_confidence = sum(
        row["self_confidence"] >= 80
        for row in failures
    )

    low_confidence = (
        len(failures) - high_confidence
    )

    labels = [
        "High confidence\n(≥80)",
        "Lower confidence\n(<80)",
    ]

    values = [
        high_confidence,
        low_confidence,
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        labels,
        values,
    )

    ax.set_ylabel(
        "Number of Perception Failures"
    )

    ax.set_title(
        "LLM Confidence on Perception Failures"
    )

    ax.set_ylim(
        0,
        max(values) + 2,
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.2,
            str(value),
            ha="center",
            va="bottom",
        )

    silent_rate = (
        high_confidence / len(failures)
        if failures
        else 0
    )

    ax.text(
        0.5,
        0.92,
        f"High-confidence failure rate: "
        f"{silent_rate:.1%}",
        transform=ax.transAxes,
        ha="center",
    )

    fig.tight_layout()

    output = (
        OUTPUT_DIR
        / "perception_failure_confidence.png"
    )

    fig.savefig(output, dpi=200)
    plt.close(fig)

    print(f"Saved: {output}")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stats = collect_stats()

    plot_accuracy(stats)
    plot_perception_failures(stats)
    plot_failure_confidence()

    print()
    print("Done.")


if __name__ == "__main__":
    main()