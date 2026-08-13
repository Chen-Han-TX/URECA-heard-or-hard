from pathlib import Path
import json
import os
import sys
import time

from anthropic import Anthropic
from dotenv import load_dotenv


load_dotenv()

client = Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

MODEL = "claude-haiku-4-5"

CONDITIONS = {"clean", "noisy", "hard"}


def build_prompt(transcript: str) -> str:
    return f"""
You are given a short question transcribed from speech.

Your job is to do ONE of two things:

1. EXTRACTION:
   If the requested value is already explicitly stated in the question
   (for example a confirmation code, room number, flight number, date,
   or amount), return that exact value.

2. REASONING:
   If the answer must be computed from the information in the question,
   compute the answer.

Examples:

Question:
What is the confirmation code K7M42?

Answer:
K7M42

Question:
What room number did they give you, 426?

Answer:
426

Question:
If you have 12 apples and give away 5, how many remain?

Answer:
7

Now answer this question:

{transcript}

Return ONLY valid JSON:

{{
  "answer": "short final answer",
  "confidence": 0
}}

Rules:
- Use only information in the transcribed question.
- Do not explain.
- Do not look up external information.
- Return the shortest possible answer.
- confidence must be an integer from 0 to 100.
- Do not include markdown.
""".strip()
    return f"""
You are answering a short question transcribed from speech.

Use ONLY the information contained in the transcribed question.

Important:
- If the question already contains the requested code, number, date,
  room number, flight number, or amount, return that value directly.
- Do not look up external information.
- For arithmetic or reasoning questions, compute the answer.
- Return the shortest answer that directly answers the question.

Question:
{transcript}

Return ONLY valid JSON:

{{
  "answer": "short final answer",
  "confidence": 0
}}

Rules:
- Do not explain.
- confidence must be an integer from 0 to 100.
- Do not include markdown.
""".strip()

def parse_response(text: str) -> dict:
    cleaned = text.strip()

    # Claude may wrap valid JSON in Markdown fences.
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        # Remove opening ```json or ```
        lines = lines[1:]

        # Remove closing ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model returned invalid JSON:\n{text}"
        ) from e

    if "answer" not in data:
        raise ValueError("Missing answer")

    if "confidence" not in data:
        raise ValueError("Missing confidence")

    confidence = int(data["confidence"])

    if not 0 <= confidence <= 100:
        raise ValueError(
            f"Invalid confidence: {confidence}"
        )

    return {
        "answer": str(data["answer"]).strip(),
        "confidence": confidence,
    }

def run_one(transcript: str) -> dict:
    prompt = build_prompt(transcript)

    start = time.perf_counter()
    ttft = None
    chunks = []

    with client.messages.stream(
        model=MODEL,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    ) as stream:

        for text in stream.text_stream:
            if ttft is None:
                ttft = time.perf_counter() - start

            chunks.append(text)

        final_message = stream.get_final_message()

    total = time.perf_counter() - start

    raw_text = "".join(chunks).strip()

    parsed = parse_response(raw_text)

    return {
        "raw_response": raw_text,
        "answer": parsed["answer"],
        "self_confidence": parsed["confidence"],
        "llm_ttft_ms": ttft * 1000 if ttft else None,
        "llm_total_ms": total * 1000,
        "input_tokens": final_message.usage.input_tokens,
        "output_tokens": final_message.usage.output_tokens,
    }


def main():
    if len(sys.argv) != 2:
        sys.exit(
            "Usage: python src/run_llm.py "
            "<clean|noisy|hard>"
        )

    condition = sys.argv[1]

    if condition not in CONDITIONS:
        sys.exit(
            "Condition must be clean, noisy, or hard"
        )

    input_path = Path(
        f"runs/asr_{condition}_labeled.jsonl"
    )

    output_path = Path(
        f"runs/llm_{condition}_haiku.jsonl"
    )

    rows = [
        json.loads(line)
        for line in input_path.read_text().splitlines()
        if line.strip()
    ]

    results = []

    for i, row in enumerate(rows, start=1):
        print(
            f"[{i}/{len(rows)}] "
            f"{condition}/{row['id']}"
        )

        result = run_one(
            row["transcript"]
        )

        combined = {
            **row,
            "llm_model": MODEL,
            **result,
        }

        results.append(combined)

        print(
            f"  answer={result['answer']!r} "
            f"conf={result['self_confidence']} "
            f"ttft={result['llm_ttft_ms']:.0f}ms"
        )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in results:
            f.write(
                json.dumps(row)
                + "\n"
            )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()