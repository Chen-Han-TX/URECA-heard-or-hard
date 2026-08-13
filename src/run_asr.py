from pathlib import Path
import json
from normalize import normalize_text 

from faster_whisper import WhisperModel
from jiwer import wer


QUESTIONS = Path("data/questions.jsonl")
AUDIO_DIR = Path("data/audio/clean")
OUTPUT = Path("runs/asr_clean.jsonl")


def load_questions():
    questions = []

    with QUESTIONS.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            questions.append(json.loads(line))

    return questions


def main():
    questions = load_questions()

    print(f"Loaded {len(questions)} questions")
    print("Loading Whisper model...")

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for i, question in enumerate(questions, start=1):
        question_id = question["id"]
        reference = question["text"]

        audio_path = AUDIO_DIR / f"{question_id}.wav"

        if not audio_path.exists():
            print(f"[SKIP] Missing {audio_path}")
            continue

        print(f"\n[{i}/{len(questions)}] {question_id}")

        segments, info = model.transcribe(
            str(audio_path),
            beam_size=5,
            language="en",
        )

        segments = list(segments)

        if not segments:
            transcript = ""
            avg_logprob = None
            no_speech_prob = None
        else:
            transcript = " ".join(
                segment.text.strip()
                for segment in segments
            )

            avg_logprob = sum(
                segment.avg_logprob
                for segment in segments
            ) / len(segments)

            no_speech_prob = max(
                segment.no_speech_prob
                for segment in segments
            )
            
        raw_wer = wer(
            reference.lower(),
            transcript.lower(),
        )

        normalized_reference = normalize_text(reference)
        normalized_transcript = normalize_text(transcript)

        normalized_wer = wer(
            normalized_reference,
            normalized_transcript,
        )
        
        result = {
            "id": question_id,
            "category": question["category"],
            "condition": "clean",

            "reference": reference,
            "transcript": transcript,

            "normalized_reference": normalized_reference,
            "normalized_transcript": normalized_transcript,

            "raw_wer": raw_wer,
            "normalized_wer": normalized_wer,

            "avg_logprob": avg_logprob,
            "no_speech_prob": no_speech_prob,
        }

        results.append(result)
        print(f"Reference : {reference}")
        print(f"Transcript: {transcript}")

        print(f"Norm ref  : {normalized_reference}")
        print(f"Norm pred : {normalized_transcript}")

        print(f"Raw WER   : {raw_wer:.3f}")
        print(f"Norm WER  : {normalized_wer:.3f}")

        if avg_logprob is not None:
            print(f"Logprob   : {avg_logprob:.3f}")

    with OUTPUT.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print()
    print("============================")
    print(f"Saved {len(results)} results")
    print(f"Output: {OUTPUT}")
    print("============================")


if __name__ == "__main__":
    main()