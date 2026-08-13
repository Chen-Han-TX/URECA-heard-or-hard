import re


NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def normalize_text(text: str) -> str:
    text = text.lower()

    # Remove common punctuation.
    text = re.sub(r"[?,.!]", "", text)

    # Remove dollar sign.
    text = text.replace("$", "")

    # Normalize ordinals such as 17th -> 17.
    text = re.sub(
        r"\b(\d+)(st|nd|rd|th)\b",
        r"\1",
        text,
    )

    # Basic number words.
    words = text.split()

    normalized_words = []

    for word in words:
        normalized_words.append(
            NUMBER_WORDS.get(word, word)
        )

    text = " ".join(normalized_words)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text