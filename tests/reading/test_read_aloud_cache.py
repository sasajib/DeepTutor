"""Timed read-aloud is memoised per page.

Narrating a page takes the local voice about a minute. A reader who stops to
re-read a sentence, or turns back a page, used to pay that minute again — and a
page turn spent it in silence with the highlight gone. The response is a pure
function of (text, voice, speed), so repeats are served from memory, and two
requests that arrive together share one narration.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from deeptutor.api.routers import reading_extensions


@pytest.fixture(autouse=True)
def _reset_cache():
    reading_extensions._timed_speech_cache.values.clear()
    reading_extensions._timed_speech_cache.inflight.clear()
    yield
    reading_extensions._timed_speech_cache.values.clear()
    reading_extensions._timed_speech_cache.inflight.clear()


def test_repeating_a_page_reuses_the_narration() -> None:
    key = reading_extensions._timed_speech_key("mat", "hello there", "af_heart", 1.0)
    cache = reading_extensions._timed_speech_cache

    async def narrate():
        # The stored timestamp is what the TTL is measured against.
        return (time.time(), {"words": []})

    async def twice():
        await cache.get_or_create(key, narrate)
        await cache.get_or_create(key, lambda: pytest.fail("narrated twice"))

    asyncio.run(twice())
    assert len(cache.values) == 1


def test_concurrent_requests_share_one_narration() -> None:
    """Double-pressing play must not narrate the same page twice."""
    produced = 0

    async def factory():
        nonlocal produced
        produced += 1
        await asyncio.sleep(0.05)
        return (time.time(), {"words": []})

    key = reading_extensions._timed_speech_key("mat", "same text", None, None)
    cache = reading_extensions._timed_speech_cache

    async def together():
        return await asyncio.gather(
            *[cache.get_or_create(key, factory) for _ in range(4)]
        )

    results = asyncio.run(together())
    assert produced == 1
    assert len(results) == 4


def test_the_key_distinguishes_voice_speed_and_text() -> None:
    base = reading_extensions._timed_speech_key("mat", "text", "af_heart", 1.0)
    assert base != reading_extensions._timed_speech_key("mat", "text", "am_puck", 1.0)
    assert base != reading_extensions._timed_speech_key("mat", "text", "af_heart", 1.25)
    assert base != reading_extensions._timed_speech_key("mat", "other", "af_heart", 1.0)
    assert base != reading_extensions._timed_speech_key("mat2", "text", "af_heart", 1.0)


def test_the_cache_is_bounded() -> None:
    """Each entry holds megabytes of audio, so it must not grow without limit."""
    assert reading_extensions._timed_speech_cache.limit <= 32


def test_a_prefetched_page_serves_the_readers_own_text_instantly() -> None:
    """The reader sends rendered text, so a prefetched page must be re-projected.

    Narrating ahead is keyed on position, but the words handed back have to be
    the ones the reader is rendering — its text differs from the stored unit by
    whitespace. Re-aligning the cached boundaries costs microseconds; narrating
    again costs a minute.
    """
    stored = {
        "audio_base64": "YXVkaW8=",
        "mime": "audio/mpeg",
        "words": [],
        "timed": True,
        # "the cost" as the voice said it.
        "boundaries": [["the", 0.0, 0.1], ["cost", 0.1, 0.2]],
    }

    rendered = "the  cost"  # same words, different whitespace
    result = reading_extensions._reproject(stored, rendered)

    assert result["audio_base64"] == "YXVkaW8="
    assert [entry["word"] for entry in result["words"]] == ["the", "cost"]
    assert [entry["start"] for entry in result["words"]] == [0.0, 0.1]
    # The client never receives the internal boundary list.
    assert "boundaries" not in result


def test_the_response_never_leaks_internal_boundaries() -> None:
    produced = {
        "audio_base64": "YXVkaW8=",
        "mime": "audio/mpeg",
        "words": [{"word": "x", "start": 0.0, "duration": 0.1}],
        "timed": True,
        "boundaries": [["x", 0.0, 0.1]],
    }

    assert "boundaries" not in reading_extensions._without_boundaries(produced)


def test_a_mismatched_cache_entry_is_rejected_rather_than_served() -> None:
    """Position and text can disagree for a moment during a page turn.

    Aligning another page's narration against this page's words collapses every
    word onto one timestamp, and the highlight then freezes for the length of the
    recording. The projection must refuse and let the page be narrated again.
    """
    # "the cost of it" as the voice said it for a different, much longer page.
    stored = {
        "audio_base64": "YXVkaW8=",
        "mime": "audio/mpeg",
        "words": [],
        "timed": True,
        "boundaries": [
            ["the", 0.0, 0.1],
            ["cost", 0.1, 0.1],
            ["of", 0.2, 0.1],
            ["it", 0.3, 0.1],
        ],
    }

    # A page of unrelated words: nothing matches, so the projection is refused.
    unrelated = " ".join(f"word{index}" for index in range(20))
    assert reading_extensions._reproject(stored, unrelated) is None


def test_a_matching_cache_entry_is_accepted() -> None:
    stored = {
        "audio_base64": "YXVkaW8=",
        "mime": "audio/mpeg",
        "words": [],
        "timed": True,
        "boundaries": [["the", 0.0, 0.1], ["cost", 0.1, 0.2]],
    }

    result = reading_extensions._reproject(stored, "the cost")

    assert result is not None
    assert [entry["start"] for entry in result["words"]] == [0.0, 0.1]


def test_a_truncated_narration_is_recognised() -> None:
    """A voice that returns part of a page must not be presented as timed speech.

    Hundreds of words sharing one timestamp is not something real speech
    produces; it is a truncated stream. Serving it freezes the highlight, so it is
    detected and the page is narrated again.
    """
    collapsed = {
        "words": [{"word": f"w{index}", "start": 0.0, "duration": 0.0} for index in range(50)]
    }
    healthy = {
        "words": [{"word": f"w{index}", "start": float(index), "duration": 0.2} for index in range(50)]
    }

    assert reading_extensions._collapsed(collapsed) is True
    assert reading_extensions._collapsed(healthy) is False
    # A short page cannot be judged this way, and must not be retried forever.
    assert reading_extensions._collapsed({"words": [{"word": "a", "start": 0.0}]}) is False
