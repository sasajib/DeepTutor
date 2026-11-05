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
