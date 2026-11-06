"""Word-timing alignment for NaturalReader-style read-along highlighting."""

from __future__ import annotations

from deeptutor.reading.word_timing import (
    TICKS_PER_SECOND,
    align_words_to_text,
)


def test_ticks_scale_matches_windows_file_time() -> None:
    # Edge reports 10,000,000 ticks per second; one second must read as 1.0.
    assert TICKS_PER_SECOND == 10_000_000


def test_every_source_token_gets_a_timing() -> None:
    text = "The **quick** brown fox jumps."
    boundaries = [
        ("The", 0.10, 0.17),
        ("quick", 0.29, 0.30),
        ("brown", 0.60, 0.30),
        ("fox", 0.91, 0.46),
        ("jumps", 1.39, 0.31),
    ]

    words = align_words_to_text(text, boundaries)

    assert [word.word for word in words] == ["The", "**quick**", "brown", "fox", "jumps."]
    assert words[0].start == 0.10
    # Markup is preserved so the client can match the rendered text.
    assert words[1].word == "**quick**"
    assert [word.start for word in words] == sorted(word.start for word in words)


def test_unmatched_boundary_does_not_shift_later_words() -> None:
    # The voice says "twelve dollars" where the page shows "$12"; alignment must
    # keep "today" anchored to its own boundary rather than drifting.
    text = "It costs $12 today."
    boundaries = [
        ("It", 0.10, 0.20),
        ("costs", 0.30, 0.30),
        ("twelve", 0.60, 0.40),
        ("dollars", 1.00, 0.40),
        ("today", 1.40, 0.40),
    ]

    words = align_words_to_text(text, boundaries)
    by_word = {word.word: word.start for word in words}

    assert by_word["It"] == 0.10
    assert by_word["today."] == 1.40


def test_trailing_tokens_still_appear() -> None:
    # A truncated boundary list must not drop visible words.
    words = align_words_to_text("one two three", [("one", 0.0, 0.2), ("two", 0.2, 0.2)])

    assert [word.word for word in words] == ["one", "two", "three"]


def test_empty_inputs_are_safe() -> None:
    assert align_words_to_text("   \n\t ", [("x", 0.0, 0.1)]) == []
    assert align_words_to_text("Hello", []) == [
        # No boundaries: the token is kept at position zero so the reader can
        # still render it, and the client falls back to its estimate.
        type(align_words_to_text("Hello", [])[0])(word="Hello", start=0.0, duration=0.0)
    ]


def test_multiple_windows_per_second() -> None:
    """Offsets are ticks, not milliseconds — a 0.5 s word is 5,000,000 ticks."""
    ticks = 5_000_000
    assert ticks / TICKS_PER_SECOND == 0.5


def test_negative_first_boundary_does_not_produce_negative_starts() -> None:
    """A provider's first word can start a hair before zero.

    Kokoro reported -0.0028s for the first word of a passage (an artifact of its
    internal alignment). The response schema rejects negative starts, so the
    endpoint answered 500 and nothing was read aloud at all. Because the previous
    word's start is carried forward, one negative value also poisoned every
    preceding word — so this must be clamped at the source, not dropped later.
    """
    text = "Preface 1. The product nobody"
    boundaries = [
        ("Preface", -0.0027916666666666923, 0.4375),
        ("one", 0.4347083333333333, 0.7375),
        (".", 1.1722083333333333, 0.15),
        ("The", 1.3222083333333332, 0.1),
        ("product", 1.4222083333333333, 0.4),
    ]

    words = align_words_to_text(text, boundaries)

    assert words, "alignment produced nothing"
    assert all(word.start >= 0 for word in words), [
        word.as_dict() for word in words
    ]
    assert all(word.duration >= 0 for word in words)
    # The words still appear in order, and the ones after the offset are intact.
    assert [word.word for word in words] == ["Preface", "1.", "The", "product", "nobody"]
    assert words[2].as_dict()["start"] == 1.322


def test_numeral_read_out_is_matched_to_its_page_token() -> None:
    """A numeral in the page must match the words the voice says for it.

    Kokoro reads "2.0" as "two". The forward matcher had no spelling for that,
    so the numeral never matched, the cursor stopped advancing, and every word
    after it fell back to one shared timestamp — 146 of 153 words collapsed onto
    a single start and the highlight sat still for the rest of the passage.
    """
    text = "shipped version 2.0 of Ledgerleaf on a Tuesday night"
    boundaries = [
        ("shipped", 0.0, 0.3),
        ("version", 0.3, 0.4),
        ("two", 0.7, 0.4),  # spoken form of "2.0"
        ("of", 1.1, 0.1),
        ("Ledgerleaf", 1.2, 0.5),
        ("on", 1.7, 0.1),
        ("a", 1.8, 0.1),
        ("Tuesday", 1.9, 0.3),
        ("night", 2.2, 0.3),
    ]

    words = align_words_to_text(text, boundaries)

    assert [word.word for word in words] == text.split()
    # Every token keeps the timestamp of the spoken word it matched.
    assert [round(word.start, 2) for word in words] == [
        0.0, 0.3, 0.7, 1.1, 1.2, 1.7, 1.8, 1.9, 2.2
    ]


def test_alignment_does_not_collapse_on_a_realistic_passage() -> None:
    """A numeral mid-passage must not freeze every later word.

    This is the end-to-end shape of the bug: one unmatched token used to make
    all following tokens inherit a single start.
    """
    text = "180 people paying 12 a month, out of about 2,200 who had ever created a free account."
    spoken = [
        ("one", 0.0, 0.2), ("hundred", 0.2, 0.2), ("eighty", 0.4, 0.3),
        ("people", 0.7, 0.3), ("paying", 1.0, 0.3), ("twelve", 1.3, 0.3),
        ("a", 1.6, 0.1), ("month", 1.7, 0.3), ("out", 2.0, 0.1),
        ("of", 2.1, 0.1), ("about", 2.2, 0.2), ("two", 2.4, 0.2),
        ("thousand", 2.6, 0.3), ("two", 2.9, 0.2), ("hundred", 3.1, 0.3),
        ("who", 3.4, 0.1), ("had", 3.5, 0.1), ("ever", 3.6, 0.1),
        ("created", 3.7, 0.3), ("a", 4.0, 0.1), ("free", 4.1, 0.3),
        ("account", 4.4, 0.4),
    ]

    words = align_words_to_text(text, spoken)

    assert len(words) == len(text.split())
    starts = [word.start for word in words]
    assert starts == sorted(starts)

    # A numeral the voice expands must take the timestamp of its first spoken
    # word, not slide down to whatever came before it.
    by_word = {word.word: word.start for word in words}
    assert by_word["180"] == 0.0, by_word["180"]
    # "2,200" starts at the same instant its first spoken word ("two") does.
    assert by_word["2,200"] == 2.2, by_word["2,200"]
    assert by_word["people"] == 0.7, by_word["people"]

    # Most tokens still get their own timing; a collapse would show as many
    # words sharing one value.
    assert len(set(starts)) >= len(starts) - 3, [
        (word.word, word.start) for word in words
    ]


def test_typographic_apostrophe_matches_the_spoken_word() -> None:
    """A curly apostrophe must not break the match.

    Python's `\\w` counts U+2019 as a word character, so "freelancer's"
    normalised to "freelancers" while the voice said "freelancer's". The mismatch
    stopped the forward cursor and cost every later word its timing: a 652-word
    page came back with 35 distinct starts covering 44s of a 214s recording.
    """
    text = "the freelancer\u2019s bank statements and Mara\u2019s report"
    boundaries = [
        ("the", 0.0, 0.1),
        ("freelancer's", 0.1, 0.4),
        ("bank", 0.5, 0.2),
        ("statements", 0.7, 0.3),
        ("and", 1.0, 0.1),
        ("Mara's", 1.1, 0.2),
        ("report", 1.3, 0.3),
    ]

    words = align_words_to_text(text, boundaries)

    assert [word.word for word in words] == text.split()
    assert [round(word.start, 2) for word in words] == [
        0.0, 0.1, 0.5, 0.7, 1.0, 1.1, 1.3
    ]


def test_a_numeral_spoken_as_several_words_keeps_the_cursor_in_step() -> None:
    """"180" is said as three words; the leftovers must not hijack later words.

    Only the first spoken word maps to the numeral's page token. If the other two
    are left in the stream, they get matched against *later* page tokens that
    happen to share a spelling — "hundred" from "one hundred eighty" steals the
    "hundred" in "two hundred", the cursor jumps past real words, and everything
    skipped over loses its own timing.

    This is the collapse that hit a 652-word page: 650 words ended up sharing 35
    timestamps covering 44s of a 214s recording.
    """
    text = "180 people and two hundred months"
    spoken = [
        ("one", 0.0, 0.1),
        ("hundred", 0.1, 0.2),
        ("eighty", 0.3, 0.2),
        ("people", 0.5, 0.3),
        ("and", 0.8, 0.1),
        ("two", 0.9, 0.2),
        ("hundred", 1.1, 0.3),
        ("months", 1.4, 0.3),
    ]

    words = align_words_to_text(text, spoken)

    assert [word.word for word in words] == text.split()
    # Each word keeps its own moment; no shared fallback timestamps.
    assert [round(word.start, 2) for word in words] == [
        0.0, 0.5, 0.8, 0.9, 1.1, 1.4
    ]
