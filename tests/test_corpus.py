import json
from pathlib import Path

import numpy as np
import pytest

from bibletts.audio import save_wav
from bibletts.books import BOOKS, book_code, parse_chapter_spec
from bibletts.config import load_config
from bibletts.corpus import (load_chapter, make_archive, parse_vpl, prepare_chapter,
                             read_manifest, resolve_splits, restore_archive, write_manifest)
from bibletts.download import catalogue, guess_book_chapter

REPO = Path(__file__).resolve().parents[1]


def test_book_table():
    assert len(BOOKS) == 66
    assert sum(n for _, _, n in BOOKS) == 1189
    assert book_code("Matthew") == "MAT" == book_code("40") == book_code("mat")
    assert book_code("1 Corinthians") == "1CO" == book_code("1corinthians")


def test_chapter_specs():
    assert parse_chapter_spec("ACT", "1-3") == [1, 2, 3]
    assert parse_chapter_spec("PSA", [23, 1]) == [1, 23]
    assert len(parse_chapter_spec("MRK", "all")) == 16
    with pytest.raises(ValueError):
        parse_chapter_spec("JUD", [2])


def test_default_splits_are_valid():
    cfg = load_config(REPO / "configs" / "data.yaml")
    splits = resolve_splits(cfg["splits"])
    assert splits[("MAT", 14)] == "dev"
    assert splits[("MAT", 13)] == "train"
    assert splits[("ACT", 22)] == "test_same"
    assert ("ROM", 1) not in splits


def test_overlapping_held_out_splits_fail():
    with pytest.raises(ValueError):
        resolve_splits({"dev": {"MAT": [1]}, "test_same": {"MAT": "1-2"}})


@pytest.mark.parametrize("name, expected", [
    ("MAT_05.mp3", ("MAT", 5)),
    ("https://example.org/audio/Matthew%2005.mp3", ("MAT", 5)),
    ("1Corinthians_13.ogg", ("1CO", 13)),
    ("1_Corinthians-13.mp3", ("1CO", 13)),
    ("40_005.mp3", ("MAT", 5)),
    ("Jude.mp3", ("JUD", 1)),
    ("readme.mp3", None),
])
def test_guess_book_chapter(name, expected):
    assert guess_book_chapter(name) == expected


def test_guess_with_regex():
    regex = r"B\d+___(?P<chapter>\d+)_(?P<book>[A-Za-z0-9]+)_"
    assert guess_book_chapter("B01___07_Matthew_____ENGWEBN2DA.mp3", regex) == ("MAT", 7)


def test_catalogue_reports_duplicates():
    with pytest.raises(ValueError):
        catalogue(["MAT_01.mp3", "Matthew-1.mp3"])
    found, unmatched = catalogue(["MAT_01.mp3", "notes.mp3"])
    assert found == {("MAT", 1): "MAT_01.mp3"} and unmatched == ["notes.mp3"]


def test_prepare_chapter_end_to_end(tmp_path):
    vpl = tmp_path / "vpl.txt"
    vpl.write_text("GEN 1:1 In the beginning, God created the heavens and the earth.\n"
                   "GEN 1:2 The earth was formless—empty.\n"
                   "TOB 1:1 A deuterocanonical verse that should be skipped.\n")
    verses = parse_vpl(vpl)
    assert list(verses) == [("GEN", 1)]

    sr = 22050
    t = np.arange(sr) / sr
    speech = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    audio = np.concatenate([speech, np.zeros(sr // 2, np.float32), speech])
    save_wav(tmp_path / "gen1.wav", audio, sr)

    cfg = load_config(REPO / "configs" / "data.yaml",
                      {"text.chapter_announcement": "{book_name}, chapter {chapter_words}."})
    out = tmp_path / "data"
    meta = prepare_chapter("GEN", 1, "test_style", tmp_path / "gen1.wav", verses[("GEN", 1)],
                           cfg, out)
    assert meta["text"].startswith("genesis, chapter one. in the beginning, god created")
    assert meta["verses"][1]["clean"] == "the earth was formless, empty."
    assert len(meta["pauses"]) == 1
    assert (out / "chapters" / "GEN_001" / "audio.wav").exists()

    rows = write_manifest(out)
    assert rows == read_manifest(out) and rows[0]["id"] == "GEN_001"
    assert load_chapter(out, "GEN", 1)["duration_s"] == pytest.approx(2.5, abs=0.01)

    archive = make_archive(out, tmp_path / "drive" / "data.tar")
    restored = restore_archive(archive, tmp_path / "restored")
    assert json.loads((restored / "chapters" / "GEN_001" / "chapter.json").read_text())["id"] == "GEN_001"
