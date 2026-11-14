"""Text cleaning and token vocabularies.

Cleaned text is a readable lowercase string such as

    "in the beginning, god created the heavens and the earth."

It contains only the alphabet's letters, single spaces, and these punctuation
marks directly after a word: , ; . ? !

The TTS model sees each punctuation mark plus its following space as one
"boundary" token, because punctuation tells it where to pause and how to
inflect. The recogniser only needs to hear words, so for it every boundary
becomes a plain space.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Characters that become a boundary, and which boundary. When several meet
# between two words, the strongest wins (see _STRENGTH).
_BOUNDARY_OF = {
    ",": ",", "—": ",", "–": ",", "(": ",", ")": ",", "[": ",", "]": ",",
    ";": ";", ":": ";",
    ".": ".", "!": "!", "?": "?",
}
_STRENGTH = {" ": 0, ",": 1, ";": 2, ".": 3, "!": 4, "?": 5}

# Marks removed without leaving a boundary. Quotes do not change how a
# sentence is read aloud enough to be worth a token.
_DROP = set("\"“”«»‹›*_")

_APOSTROPHES = "'‘’ʼ`´"

# ---------------------------------------------------------------------------
# Numbers to words. The World English Bible mostly writes numbers as words,
# so this only needs to cover what slips through.
# ---------------------------------------------------------------------------

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve "
         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = [(10**9, "billion"), (10**6, "million"), (1000, "thousand"), (100, "hundred")]


def number_to_words(n: int) -> str:
    """Cardinal number in words, British-style 'and': 120 -> one hundred and twenty."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ("-" + _ONES[ones] if ones else "")
    for value, name in _SCALES:
        if n >= value:
            head, rest = divmod(n, value)
            words = number_to_words(head) + " " + name
            if rest:
                words += (" and " if rest < 100 else " ") + number_to_words(rest)
            return words
    raise AssertionError("unreachable")


_ORDINAL_FIX = {"one": "first", "two": "second", "three": "third", "five": "fifth",
                "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


def ordinal_to_words(n: int) -> str:
    words = number_to_words(n)
    head, _, last = words.rpartition(" ")
    if "-" in last:
        h2, _, last = last.rpartition("-")
        head = (head + " " if head else "") + h2 + "-"
        sep = ""
    else:
        sep = " " if head else ""
    if last in _ORDINAL_FIX:
        last = _ORDINAL_FIX[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + sep + last


def _spell_numbers(text: str) -> str:
    text = re.sub(r"(\d+)(st|nd|rd|th)\b", lambda m: ordinal_to_words(int(m.group(1))), text)
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)  # 1,000 -> 1000
    return re.sub(r"\d+", lambda m: number_to_words(int(m.group(0))), text)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def normalise_unicode(text: str, strip_accents: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text)
    for a in _APOSTROPHES:
        text = text.replace(a, "'")
    if strip_accents:
        decomposed = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in decomposed if not unicodedata.combining(c))
    return text


def clean_text(
    raw: str,
    alphabet: str = "abcdefghijklmnopqrstuvwxyz'",
    strip_accents: bool = True,
    dropped: Counter | None = None,
) -> str:
    """Turn raw verse text into cleaned text (see module docstring).

    If `dropped` is given, every unexpected character that was removed is
    counted in it, so the caller can check nothing important is being lost.
    """
    text = normalise_unicode(raw, strip_accents).lower()
    text = _spell_numbers(text)
    letters = set(alphabet) - {"'"}

    # Walk the text once, splitting it into words and the gaps between them.
    words: list[str] = []
    gaps: list[str] = []  # gaps[i] is the boundary after words[i]
    word = ""
    gap = " "
    for ch in text:
        if ch in letters or (ch == "'" and "'" in alphabet):
            if not word:  # a new word starts
                if words:
                    gaps.append(gap)
                gap = " "  # punctuation before the first word is ignored
            word += ch
            continue
        if word:
            words.append(word)
            word = ""
        if ch.isspace() or ch in _DROP or ch == "-":
            continue  # hyphens join compound words; read as a plain gap
        if ch in _BOUNDARY_OF:
            b = _BOUNDARY_OF[ch]
            if _STRENGTH[b] > _STRENGTH[gap]:
                gap = b
            continue
        if dropped is not None:
            dropped[ch] += 1
    if word:
        words.append(word)
    gaps.append(gap)  # boundary after the last word

    # Apostrophes used as quote marks: drop them at word edges, except a
    # trailing one after s, which marks a plural possessive ("disciples'").
    tidy = []
    for w in words:
        w = w.lstrip("'")
        if w.endswith("'") and not w.endswith("s'"):
            w = w.rstrip("'")
        tidy.append(w)

    keep = [i for i, w in enumerate(tidy) if w]
    if not keep:
        return ""
    out = []
    for n, i in enumerate(keep):
        out.append(tidy[i])
        # Strongest boundary among the gaps between this word and the next
        # kept word (or the end of the text).
        end = keep[n + 1] if n + 1 < len(keep) else len(gaps)
        b = max(gaps[i:end], key=_STRENGTH.get)
        out.append(b + " " if b != " " else " ")
    return "".join(out).rstrip(" ")


def for_recogniser(cleaned: str) -> str:
    """Remove punctuation, leaving words separated by single spaces."""
    return re.sub(r"[,;.?!]", "", cleaned).strip()


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

BOUNDARIES = [" ", ", ", "; ", ". ", "? ", "! "]


class TTSVocab:
    """Tokens for the acoustic model: letters plus typed word boundaries.

    A punctuation mark and the space after it are one token. Punctuation at
    the very end of a text (with no space after it) maps to the same token.
    """

    def __init__(self, alphabet: str):
        self.specials = ["<pad>", "<bos>", "<eos>"]
        self.tokens = self.specials + BOUNDARIES + list(alphabet)
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.pad, self.bos, self.eos = 0, 1, 2

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, cleaned: str, add_bos_eos: bool = True) -> list[int]:
        pieces = re.findall(r"[,;.?!] ?| |.", cleaned)
        ids = []
        for p in pieces:
            if p in ",;.?!":
                p = p + " "
            ids.append(self.index[p])
        return [self.bos, *ids, self.eos] if add_bos_eos else ids

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.tokens[i] for i in ids if i >= len(self.specials))
        return text.rstrip(" ")


class CTCVocab:
    """Tokens for the recogniser: blank, space, letters."""

    def __init__(self, alphabet: str):
        self.tokens = ["<blank>", " "] + list(alphabet)
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.blank = 0

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, cleaned: str) -> list[int]:
        return [self.index[c] for c in for_recogniser(cleaned)]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.tokens[i] for i in ids if i != self.blank)
