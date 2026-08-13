from pathlib import Path
import sys

from faster_whisper import WhisperModel


def transcribe_audio(audio_path: str) -> None:
    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    print("Loading Whisper model...")
    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
    )

    print(f"Transcribing: {path}")

    segments, info = model.transcribe(
        str(path),
        beam_size=5,
        language="en",
    )

    segments = list(segments)

    if not segments:
        print("No speech detected.")
        return

    transcript = " ".join(segment.text.strip() for segment in segments)

    avg_logprob = sum(
        segment.avg_logprob for segment in segments
    ) / len(segments)

    max_no_speech_prob = max(
        segment.no_speech_prob for segment in segments
    )

    print()
    print("========== RESULT ==========")
    print(f"Language: {info.language}")
    print(f"Transcript: {transcript}")
    print(f"Average log probability: {avg_logprob:.4f}")
    print(f"Max no-speech probability: {max_no_speech_prob:.4f}")
    print("============================")
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python src/transcribe.py <audio_file>")

    transcribe_audio(sys.argv[1])