import re

from word2number import w2n


NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "and",
}


ORDINAL_WORDS = {
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
    "eleventh": "11",
    "twelfth": "12",
    "thirteenth": "13",
    "fourteenth": "14",
    "fifteenth": "15",
    "sixteenth": "16",
    "seventeenth": "17",
    "eighteenth": "18",
    "nineteenth": "19",
    "twentieth": "20",
}


def words_to_number(phrase: str):
    """
    Convert an English number phrase to a number.

    Returns None if conversion fails.
    """
    phrase = phrase.strip()

    try:
        return w2n.word_to_num(phrase)
    except ValueError:
        return None


def normalize_money_phrases(text: str) -> str:
    """
    Normalize money expressions.

    Examples:
      twenty seven dollars and fifty cents -> 27.50
      four dollars ninety                  -> 4.90
      four dollars                         -> 4
    """

    number_pattern = (
        r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
        r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
        r"eighty|ninety|hundred|thousand|and)"
    )

    # Case 1:
    # "twenty seven dollars and fifty cents"
    dollars_and_cents_pattern = re.compile(
        rf"\b((?:{number_pattern}\s*)+?)"
        rf"\s+dollars\s+and\s+"
        rf"((?:{number_pattern}\s*)+?)"
        rf"\s+cents\b"
    )

    def replace_dollars_and_cents(match):
        dollars = words_to_number(match.group(1))
        cents = words_to_number(match.group(2))

        if dollars is None or cents is None:
            return match.group(0)

        return f"{dollars}.{cents:02d}"

    text = dollars_and_cents_pattern.sub(
        replace_dollars_and_cents,
        text,
    )

    # Case 2:
    # "four dollars ninety"
    dollars_with_bare_cents_pattern = re.compile(
        rf"\b((?:{number_pattern}\s*)+?)"
        rf"\s+dollars\s+"
        rf"((?:{number_pattern}\s*)+?)"
        rf"(?=\s|$)"
    )

    def replace_dollars_with_bare_cents(match):
        dollars = words_to_number(match.group(1))
        cents = words_to_number(match.group(2))

        if dollars is None or cents is None:
            return match.group(0)

        # Only treat the second number as cents if it makes sense.
        if not 0 <= cents <= 99:
            return match.group(0)

        return f"{dollars}.{cents:02d}"

    text = dollars_with_bare_cents_pattern.sub(
        replace_dollars_with_bare_cents,
        text,
    )

    # Case 3:
    # "four dollars"
    dollars_only_pattern = re.compile(
        rf"\b((?:{number_pattern}\s*)+?)"
        rf"\s+dollars\b"
    )

    def replace_dollars_only(match):
        dollars = words_to_number(match.group(1))

        if dollars is None:
            return match.group(0)

        return str(dollars)

    text = dollars_only_pattern.sub(
        replace_dollars_only,
        text,
    )

    return text

def normalize_number_phrases(text: str) -> str:
    """
    Convert continuous English number phrases into digits.

    Examples:
      'four hundred and twenty six' -> '426'
      'two and arrives'             -> '2 and arrives'

    'and' is treated as part of a number only when it is surrounded
    by number words.
    """

    tokens = text.split()
    output = []

    current_number_tokens = []

    def flush_number():
        nonlocal current_number_tokens

        if not current_number_tokens:
            return

        phrase = " ".join(current_number_tokens)

        try:
            value = w2n.word_to_num(phrase)
            output.append(str(value))
        except ValueError:
            output.extend(current_number_tokens)

        current_number_tokens = []

    for i, token in enumerate(tokens):
        if token == "and":
            previous_is_number = (
                bool(current_number_tokens)
            )

            next_is_number = (
                i + 1 < len(tokens)
                and tokens[i + 1] in NUMBER_WORDS
                and tokens[i + 1] != "and"
            )

            # Only keep "and" inside a number phrase when it connects
            # two numeric components:
            #
            # "four hundred and twenty six"
            #
            # but NOT:
            #
            # "two and arrives"
            if previous_is_number and next_is_number:
                current_number_tokens.append(token)
            else:
                flush_number()
                output.append(token)

        elif token in NUMBER_WORDS:
            current_number_tokens.append(token)

        else:
            flush_number()
            output.append(token)

    flush_number()

    return " ".join(output)

def normalize_text(text: str) -> str:
    text = text.lower()

    # Whisper frequently emits "$4.90"; treat $ as formatting.
    text = text.replace("$", "")

    # 17th -> 17
    text = re.sub(
        r"\b(\d+)(st|nd|rd|th)\b",
        r"\1",
        text,
    )

    # seventeenth -> 17
    for word, number in ORDINAL_WORDS.items():
        text = re.sub(
            rf"\b{word}\b",
            number,
            text,
        )

    # Remove punctuation, but preserve decimal point.
    text = re.sub(r"[?,!]", " ", text)
    # Remove periods unless they are decimal points between digits.
    text = re.sub(r"(?<!\d)\.", " ", text)
    text = re.sub(r"\.(?!\d)", " ", text)

    # Money must be normalized before generic numbers.
    text = normalize_money_phrases(text)

    # Remaining number phrases.
    text = normalize_number_phrases(text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text