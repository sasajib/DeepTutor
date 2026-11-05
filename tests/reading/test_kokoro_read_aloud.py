"""Kokoro read-along: exact word timings, and no Markdown read aloud.

These cover the two behaviours that only show up against a real Kokoro server:
the captioned endpoint's timestamps reaching the reader, and the fact that Kokoro
pronounces Markdown structure literally ("###" becomes "number number number").
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from deeptutor.reading import word_timing
from deeptutor.services.voice.config import TTSConfig


def _config(**overrides) -> TTSConfig:
    base = TTSConfig(
        model="kokoro",
        provider_name="kokoro",
        adapter="kokoro",
        base_url="http://kokoro.test",
        voice="af_heart",
        response_format="mp3",
        request_timeout=30,
    )
    return replace(base, **overrides) if overrides else base


def test_timed_speech_uses_kokoro_exact_words(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_synth(text: str, config: TTSConfig):
        calls.append(text)
        return b"audio-bytes", "audio/mpeg", [
            {"word": "Preface", "start": 0.0, "duration": 0.4},
            {"word": "hello", "start": 0.4, "duration": 0.3},
        ]

    monkeypatch.setattr(
        "deeptutor.services.voice.adapters.kokoro.synthesize_kokoro_with_timings",
        fake_synth,
    )
    monkeypatch.setattr(
        word_timing, "resolve_tts_runtime_config", lambda: _config(), raising=False
    )
    # `resolve_tts_runtime_config` is imported inside the function, so patch the
    # module it is imported from.
    monkeypatch.setattr(
        "deeptutor.services.config.provider_runtime.resolve_tts_runtime_config",
        lambda: _config(),
    )

    result = asyncio.run(word_timing.synthesize_with_word_timings("Preface hello"))

    assert result["timed"] is True
    assert [word["word"] for word in result["words"]] == ["Preface", "hello"]
    assert result["audio_base64"] == "YXVkaW8tYnl0ZXM="


def test_markdown_is_not_read_aloud(monkeypatch) -> None:
    """A heading must not be pronounced as three copies of "number"."""
    spoken: list[str] = []

    async def fake_synth(text: str, config: TTSConfig):
        spoken.append(text)
        return b"x", "audio/mpeg", []

    monkeypatch.setattr(
        "deeptutor.services.voice.adapters.kokoro.synthesize_kokoro_with_timings",
        fake_synth,
    )
    monkeypatch.setattr(
        "deeptutor.services.config.provider_runtime.resolve_tts_runtime_config",
        lambda: _config(),
    )

    asyncio.run(
        word_timing.synthesize_with_word_timings(
            "### Who this book is for\n\n**Bold** and `code` live here."
        )
    )

    assert spoken, "nothing was synthesised"
    heard = spoken[0]
    assert "###" not in heard
    assert "**" not in heard
    assert "`" not in heard
    assert "Who this book is for" in heard


def test_markup_only_unit_reports_untimed_rather_than_failing(monkeypatch) -> None:
    async def fake_synth(text: str, config: TTSConfig):  # pragma: no cover
        raise AssertionError("a unit with no speakable text must not be synthesised")

    monkeypatch.setattr(
        "deeptutor.services.voice.adapters.kokoro.synthesize_kokoro_with_timings",
        fake_synth,
    )
    monkeypatch.setattr(
        "deeptutor.services.config.provider_runtime.resolve_tts_runtime_config",
        lambda: _config(),
    )

    result = asyncio.run(word_timing.synthesize_with_word_timings("---\n\n| | |\n"))

    assert result["timed"] is False
    assert result["words"] == []


def test_kokoro_is_a_registered_provider_and_adapter() -> None:
    from deeptutor.services.config.provider_runtime import TTS_PROVIDERS
    from deeptutor.services.voice.adapters import TTS_ADAPTERS

    spec = TTS_PROVIDERS["kokoro"]
    assert spec.adapter == "kokoro"
    assert spec.is_local is True
    assert spec.default_voice == "af_heart"
    assert "kokoro" in TTS_ADAPTERS


def test_reader_supplied_text_keeps_its_own_word_numbering(monkeypatch) -> None:
    """The highlight indexes the page's word list, so marker words must survive.

    A rendered list item still starts with "1." and the page counts it as a word.
    Stripping it server-side removed a slot from the audio's word list while the
    page kept one, so the highlight ran ahead of the voice from the first list
    item onward — the exact "reading is behind, highlight is forward" report.
    """
    spoken: list[str] = []

    async def fake_synth(text: str, config: TTSConfig):
        spoken.append(text)
        return b"x", "audio/mpeg", [{"word": "x", "start": 0.0, "duration": 0.1}]

    monkeypatch.setattr(
        "deeptutor.services.voice.adapters.kokoro.synthesize_kokoro_with_timings",
        fake_synth,
    )
    monkeypatch.setattr(
        "deeptutor.services.config.provider_runtime.resolve_tts_runtime_config",
        lambda: _config(),
    )

    rendered = "1. The product nobody is looking for"
    asyncio.run(
        word_timing.synthesize_with_word_timings(rendered, strip_markdown=False)
    )

    assert spoken, "nothing was synthesised"
    # The marker stays, so the token count matches what is on screen.
    assert spoken[0].split()[0] == "1."
    assert len(spoken[0].split()) == len(rendered.split())


def test_unit_text_is_still_stripped_by_default(monkeypatch) -> None:
    """Text read from the store is Markdown and must be stripped."""
    spoken: list[str] = []

    async def fake_synth(text: str, config: TTSConfig):
        spoken.append(text)
        return b"x", "audio/mpeg", []

    monkeypatch.setattr(
        "deeptutor.services.voice.adapters.kokoro.synthesize_kokoro_with_timings",
        fake_synth,
    )
    monkeypatch.setattr(
        "deeptutor.services.config.provider_runtime.resolve_tts_runtime_config",
        lambda: _config(),
    )

    asyncio.run(word_timing.synthesize_with_word_timings("### Heading here"))
    assert spoken[0].strip() == "Heading here"


def test_kokoro_timestamps_are_aligned_to_the_page_not_the_voice(monkeypatch) -> None:
    """Kokoro's spoken tokens are not the page's tokens.

    It expands a list marker into a word plus punctuation ("1." -> "one" + "."),
    so the raw timestamp list is *longer* than the page's word list. Returning
    those directly made the highlight index past the end of the page's words and
    run ahead of the voice. They must be projected onto the page's own tokens.
    """
    rendered = "1. The product nobody is looking for"

    async def fake_synth(text: str, config: TTSConfig):
        # Spoken tokens: the marker became "one" + ".", so one more than the page.
        spoken = [
            ("one", 0.0, 0.2),
            (".", 0.2, 0.1),
            ("The", 0.3, 0.2),
            ("product", 0.5, 0.3),
            ("nobody", 0.8, 0.3),
            ("is", 1.1, 0.1),
            ("looking", 1.2, 0.3),
            ("for", 1.5, 0.2),
        ]
        return b"x", "audio/mpeg", [
            {"word": word, "start": start, "duration": duration}
            for word, start, duration in spoken
        ]

    monkeypatch.setattr(
        "deeptutor.services.voice.adapters.kokoro.synthesize_kokoro_with_timings",
        fake_synth,
    )
    monkeypatch.setattr(
        "deeptutor.services.config.provider_runtime.resolve_tts_runtime_config",
        lambda: _config(),
    )

    result = asyncio.run(
        word_timing.synthesize_with_word_timings(rendered, strip_markdown=False)
    )

    page_tokens = rendered.split()
    returned = [word["word"] for word in result["words"]]
    # One timing per word on the page, in the page's order — no extra slots.
    assert returned == page_tokens
    assert len(result["words"]) == len(page_tokens)
