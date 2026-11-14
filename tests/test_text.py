from collections import Counter

import pytest

from bibletts.text import (CTCVocab, TTSVocab, clean_text, for_recogniser,
                           number_to_words, ordinal_to_words)


@pytest.mark.parametrize("n, words", [
    (0, "zero"), (7, "seven"), (19, "nineteen"), (21, "twenty-one"), (100, "one hundred"),
    (120, "one hundred and twenty"), (969, "nine hundred and sixty-nine"),
    (1000, "one thousand"), (3005, "three thousand and five"),
    (144000, "one hundred and forty-four thousand"),
])
def test_number_to_words(n, words):
    assert number_to_words(n) == words


@pytest.mark.parametrize("n, words", [
    (1, "first"), (2, "second"), (3, "third"), (12, "twelfth"), (20, "twentieth"),
    (21, "twenty-first"), (100, "one hundredth"),
])
def test_ordinals(n, words):
    assert ordinal_to_words(n) == words


@pytest.mark.parametrize("raw, clean", [
    ("In the beginning, God created the heavens and the earth.",
     "in the beginning, god created the heavens and the earth."),
    # Curly quotes dropped, question kept, strongest boundary wins.
    ("He said, “Who are you?”", "he said, who are you?"),
    # Em dash and parentheses become commas; colon becomes semicolon.
    ("Jesus—the Christ (the anointed one): said", "jesus, the christ, the anointed one; said"),
    # Apostrophes: contractions and plural possessives kept, quote marks dropped.
    ("‘don’t touch the disciples’ bread’", "don't touch the disciples' bread"),
    # Numbers spelled out, hyphenated compounds read as separate words.
    ("about 3,000 souls were well-pleased", "about three thousand souls were well pleased"),
    ("the 3rd day", "the third day"),
    # Accents stripped, leading punctuation ignored, verse with no end mark.
    ("...Zaécheus and the", "zaecheus and the"),
    ("", ""),
])
def test_clean_text(raw, clean):
    assert clean_text(raw) == clean


def test_unexpected_characters_are_counted():
    dropped = Counter()
    assert clean_text("alpha § omega #", dropped=dropped) == "alpha omega"
    assert dropped == Counter({"§": 1, "#": 1})


def test_for_recogniser():
    assert for_recogniser("he said, who are you?") == "he said who are you"


def test_tts_vocab_round_trip():
    vocab = TTSVocab("abcdefghijklmnopqrstuvwxyz'")
    text = "and he said, let there be light. and there was light!"
    ids = vocab.encode(text)
    assert ids[0] == vocab.bos and ids[-1] == vocab.eos
    assert vocab.decode(ids) == text
    # ", " is one token, not two.
    assert len(vocab.encode("a, b", add_bos_eos=False)) == 3


def test_ctc_vocab_round_trip():
    vocab = CTCVocab("abcdefghijklmnopqrstuvwxyz'")
    ids = vocab.encode("it's light, now.")
    assert vocab.blank not in ids
    assert vocab.decode(ids) == "it's light now"
