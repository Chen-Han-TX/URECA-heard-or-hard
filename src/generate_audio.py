from pathlib import Path
import json
import subprocess


QUESTIONS = Path("data/questions.jsonl")
OUTPUT_DIR = Path("data/audio/clean")


def run_command(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with QUESTIONS.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            question = json.loads(line)

            question_id = question["id"]
            text = question["text"]

            aiff_path = OUTPUT_DIR / f"{question_id}.aiff"
            wav_path = OUTPUT_DIR / f"{question_id}.wav"

            print(f"\nGenerating {question_id}...")

            run_command([
                "say",
                "-v",
                "Samantha",
                "-o",
                str(aiff_path),
                text,
            ])

            run_command([
                "ffmpeg",
                "-y",
                "-i",
                str(aiff_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                str(wav_path),
            ])

            aiff_path.unlink()


if __name__ == "__main__":
    main()