"""The 66 books of the Protestant canon: USFM codes, names, chapter counts.

USFM codes (GEN, MAT, 1CH, ...) are the standard identifiers used by Bible
text files, including eBible's verse-per-line format.
"""

from __future__ import annotations

import re

# (code, English name, number of chapters), in canonical order.
BOOKS: list[tuple[str, str, int]] = [
    ("GEN", "Genesis", 50), ("EXO", "Exodus", 40), ("LEV", "Leviticus", 27),
    ("NUM", "Numbers", 36), ("DEU", "Deuteronomy", 34), ("JOS", "Joshua", 24),
    ("JDG", "Judges", 21), ("RUT", "Ruth", 4), ("1SA", "1 Samuel", 31),
    ("2SA", "2 Samuel", 24), ("1KI", "1 Kings", 22), ("2KI", "2 Kings", 25),
    ("1CH", "1 Chronicles", 29), ("2CH", "2 Chronicles", 36), ("EZR", "Ezra", 10),
    ("NEH", "Nehemiah", 13), ("EST", "Esther", 10), ("JOB", "Job", 42),
    ("PSA", "Psalms", 150), ("PRO", "Proverbs", 31), ("ECC", "Ecclesiastes", 12),
    ("SNG", "Song of Solomon", 8), ("ISA", "Isaiah", 66), ("JER", "Jeremiah", 52),
    ("LAM", "Lamentations", 5), ("EZK", "Ezekiel", 48), ("DAN", "Daniel", 12),
    ("HOS", "Hosea", 14), ("JOL", "Joel", 3), ("AMO", "Amos", 9),
    ("OBA", "Obadiah", 1), ("JON", "Jonah", 4), ("MIC", "Micah", 7),
    ("NAM", "Nahum", 3), ("HAB", "Habakkuk", 3), ("ZEP", "Zephaniah", 3),
    ("HAG", "Haggai", 2), ("ZEC", "Zechariah", 14), ("MAL", "Malachi", 4),
    ("MAT", "Matthew", 28), ("MRK", "Mark", 16), ("LUK", "Luke", 24),
    ("JHN", "John", 21), ("ACT", "Acts", 28), ("ROM", "Romans", 16),
    ("1CO", "1 Corinthians", 16), ("2CO", "2 Corinthians", 13),
    ("GAL", "Galatians", 6), ("EPH", "Ephesians", 6), ("PHP", "Philippians", 4),
    ("COL", "Colossians", 4), ("1TH", "1 Thessalonians", 5),
    ("2TH", "2 Thessalonians", 3), ("1TI", "1 Timothy", 6), ("2TI", "2 Timothy", 4),
    ("TIT", "Titus", 3), ("PHM", "Philemon", 1), ("HEB", "Hebrews", 13),
    ("JAS", "James", 5), ("1PE", "1 Peter", 5), ("2PE", "2 Peter", 3),
    ("1JN", "1 John", 5), ("2JN", "2 John", 1), ("3JN", "3 John", 1),
    ("JUD", "Jude", 1), ("REV", "Revelation", 22),
]

CODES = [code for code, _, _ in BOOKS]
NAME = {code: name for code, name, _ in BOOKS}
CHAPTERS = {code: n for code, _, n in BOOKS}
NUMBER = {code: i + 1 for i, code in enumerate(CODES)}  # GEN = 1 ... REV = 66


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# Lookup from many spellings to a code: "MAT", "matthew", "1corinthians",
# "1cor", plus a few common alternatives.
_ALIASES: dict[str, str] = {}
for _code, _name, _ in BOOKS:
    _ALIASES[_squash(_code)] = _code
    _ALIASES[_squash(_name)] = _code
_ALIASES.update({
    "psalm": "PSA", "song": "SNG", "songofsongs": "SNG", "canticles": "SNG",
    "revelations": "REV", "acts": "ACT", "actsoftheapostles": "ACT",
    "mark": "MRK", "john": "JHN", "judges": "JDG", "jude": "JUD",
    "ezekiel": "EZK", "philippians": "PHP", "philemon": "PHM",
})


def book_code(token: str) -> str | None:
    """Turn a book code, name or number (1-66) into a USFM code, or None."""
    squashed = _squash(token)
    if squashed.isdigit():
        n = int(squashed)
        return CODES[n - 1] if 1 <= n <= len(CODES) else None
    return _ALIASES.get(squashed)


def parse_chapter_spec(code: str, spec: str | int | list[int]) -> list[int]:
    """Turn "all", "1-21", 5 or [1, 2, 3] into a sorted list of chapters."""
    total = CHAPTERS[code]
    if spec == "all":
        chapters = list(range(1, total + 1))
    elif isinstance(spec, int):
        chapters = [spec]
    elif isinstance(spec, str):
        chapters = []
        for part in spec.split(","):
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-"))
                chapters.extend(range(lo, hi + 1))
            else:
                chapters.append(int(part))
    else:
        chapters = [int(c) for c in spec]
    bad = [c for c in chapters if not 1 <= c <= total]
    if bad:
        raise ValueError(f"{code} has {total} chapters; got {bad}")
    return sorted(set(chapters))
