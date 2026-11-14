"""Building the chapter-level corpus: text, splits, prepared audio, manifest.

The output of milestone 1 is one folder per chapter set up like this:

    <data_root>/chapters/MAT_005/
        audio.wav       mono, config sample rate, loudness-normalised
        chapter.json    book, chapter, split, duration, verses (raw and
                        cleaned), the full cleaned text, detected pauses

plus <data_root>/manifest.jsonl with one line per chapter.
"""

from __future__ import annotations

import json
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from .audio import detect_pauses, load_audio, normalise_loudness, save_wav
from .books import CHAPTERS, CODES, NAME, parse_chapter_spec
from .text import clean_text, number_to_words

ChapterKey = tuple[str, int]

# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

_VPL_LINE = re.compile(r"^([1-3]?[A-Z]{2,3})\s+(\d+):(\d+)\s+(.*)$")


def parse_vpl(path: str | Path) -> dict[ChapterKey, list[tuple[int, str]]]:
    """Read a verse-per-line file ("MAT 5:3 Blessed are...").

    Returns {(book, chapter): [(verse, raw text), ...]} for the 66 standard
    books. Other books (for example deuterocanonical ones) are skipped.
    """
    chapters: dict[ChapterKey, list[tuple[int, str]]] = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            m = _VPL_LINE.match(line.strip())
            if not m:
                continue
            book, chapter, verse, text = m.groups()
            if book in CHAPTERS and text.strip():
                chapters[(book, int(chapter))].append((int(verse), text.strip()))
    return dict(chapters)


def announcement_text(template: str | None, book: str, chapter: int) -> str | None:
    """Raw text of the spoken chapter announcement, if the reader says one."""
    if not template:
        return None
    return template.format(book_name=NAME[book], chapter_words=number_to_words(chapter))


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def resolve_splits(split_cfg: dict) -> dict[ChapterKey, str]:
    """{(book, chapter): split name}. Fails if a chapter is in two splits.

    A chapter listed under "dev" is removed from "train" automatically, so
    "MAT: all" in train plus "MAT: [14]" in dev means train gets the rest.
    """
    assignment: dict[ChapterKey, str] = {}
    held_out_first = sorted(split_cfg, key=lambda s: s == "train")  # train last
    for split in held_out_first:
        for book, spec in split_cfg[split].items():
            for ch in parse_chapter_spec(book, spec):
                key = (book, ch)
                if key in assignment:
                    if split == "train":
                        continue  # already held out
                    raise ValueError(f"{book} {ch} is in both {assignment[key]} and {split}")
                assignment[key] = split
    order = {code: i for i, code in enumerate(CODES)}
    return dict(sorted(assignment.items(), key=lambda kv: (order[kv[0][0]], kv[0][1])))


# ---------------------------------------------------------------------------
# Preparing chapters
# ---------------------------------------------------------------------------


def chapter_id(book: str, chapter: int) -> str:
    return f"{book}_{chapter:03d}"


def prepare_chapter(book: str, chapter: int, split: str, audio_path: str | Path,
                    verses: list[tuple[int, str]], cfg: dict, out_root: str | Path,
                    dropped: Counter | None = None, overwrite: bool = False) -> dict:
    """Convert one chapter's audio and text into the milestone-1 format."""
    out_dir = Path(out_root) / "chapters" / chapter_id(book, chapter)
    meta_path = out_dir / "chapter.json"
    if meta_path.exists() and not overwrite:
        return json.loads(meta_path.read_text())

    a, t, p = cfg["audio"], cfg["text"], cfg["pauses"]
    sr = a["sample_rate"]
    audio = load_audio(audio_path, sr)
    audio = normalise_loudness(audio, sr, a["loudness_target_dbfs"])
    save_wav(out_dir / "audio.wav", audio, sr)

    clean = lambda raw: clean_text(raw, t["alphabet"], t["strip_accents"], dropped)  # noqa: E731
    verse_records = [{"verse": v, "raw": raw, "clean": clean(raw)} for v, raw in verses]
    announcement = announcement_text(t.get("chapter_announcement"), book, chapter)
    pieces = ([clean(announcement)] if announcement else []) + [v["clean"] for v in verse_records]

    meta = {
        "id": chapter_id(book, chapter),
        "book": book,
        "chapter": chapter,
        "split": split,
        "sample_rate": sr,
        "duration_s": round(len(audio) / sr, 3),
        "source_audio": str(audio_path),
        "announcement": clean(announcement) if announcement else None,
        "verses": verse_records,
        "text": " ".join(x for x in pieces if x),
        "pauses": detect_pauses(audio, sr, p["threshold_db"], p["min_duration_s"]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return meta


def write_manifest(out_root: str | Path) -> list[dict]:
    """Collect every chapter.json into manifest.jsonl (without verse details)."""
    out_root = Path(out_root)
    rows = []
    for meta_path in sorted((out_root / "chapters").glob("*/chapter.json")):
        m = json.loads(meta_path.read_text())
        rows.append({k: m[k] for k in ("id", "book", "chapter", "split", "duration_s")}
                    | {"n_verses": len(m["verses"]), "n_chars": len(m["text"]),
                       "n_pauses": len(m["pauses"])})
    with open(out_root / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def read_manifest(out_root: str | Path) -> list[dict]:
    with open(Path(out_root) / "manifest.jsonl") as f:
        return [json.loads(line) for line in f]


def load_chapter(out_root: str | Path, book: str, chapter: int) -> dict:
    path = Path(out_root) / "chapters" / chapter_id(book, chapter) / "chapter.json"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Archiving to and from Google Drive
# ---------------------------------------------------------------------------


def make_archive(src_dir: str | Path, archive_path: str | Path) -> Path:
    """Pack a folder into one .tar file. Drive handles one big file far better
    than thousands of small ones. No gzip: WAV barely compresses with it and
    it would add minutes."""
    src_dir, archive_path = Path(src_dir), Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = archive_path.with_suffix(".tar.part")
    with tarfile.open(tmp, "w") as tar:
        tar.add(src_dir, arcname=src_dir.name)
    tmp.rename(archive_path)
    return archive_path


def restore_archive(archive_path: str | Path, dest_parent: str | Path) -> Path:
    """Unpack an archive made by make_archive into dest_parent."""
    dest_parent = Path(dest_parent)
    dest_parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as tar:
        try:
            tar.extractall(dest_parent, filter="data")
        except TypeError:  # Python without extraction filters
            tar.extractall(dest_parent)
        top = tar.getnames()[0].split("/")[0]
    return dest_parent / top
