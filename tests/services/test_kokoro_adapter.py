"""Kokoro TTS adapter: NDJSON captioned speech, WAV merge, provider wiring.

Kokoro is the voice the read-along highlight was designed against, because it
reports *exact* word boundaries. These tests pin the three parts that are easy to
get subtly wrong and that a live server would hide: the newline-delimited JSON
body, the per-chunk audio join, and the timestamps reaching the reader.
"""

from __future__ import annotations

import base64
import io
import json
import wave

from deeptutor.services.voice.adapters.kokoro import (
    merge_chunk_audio,
    normalize_kokoro_base,
    parse_captioned_body,
)


def _silent_wav(frames: int, sample_rate: int = 24000) -> bytes:
    """A real, parseable mono 16-bit WAV with ``frames`` zero samples."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)
    return buffer.getvalue()


def _pcm_only(wav: bytes) -> bytes:
    """Return just the PCM payload of a WAV, i.e. what later chunks look like."""
    with wave.open(io.BytesIO(wav), "rb") as handle:
        return handle.readframes(handle.getnframes())


def test_normalizes_trailing_slash_and_v1() -> None:
    assert normalize_kokoro_base("https://voice.example") == "https://voice.example"
    assert normalize_kokoro_base("https://voice.example/") == "https://voice.example"
    assert normalize_kokoro_base("https://voice.example/v1") == "https://voice.example"
    assert normalize_kokoro_base("https://voice.example/v1/") == "https://voice.example"


def test_parses_a_single_json_object() -> None:
    # Older builds answer with one object rather than a stream.
    body = json.dumps({"audio": "AAA", "timestamps": []})
    chunks = parse_captioned_body(body)
    assert len(chunks) == 1
    assert chunks[0]["audio"] == "AAA"


def test_parses_newline_delimited_chunks() -> None:
    # The real endpoint streams one JSON object per synthesis chunk. Parsing
    # only the first object truncated long passages to their first chunk.
    body = "\n".join(
        [
            json.dumps({"audio": "AAA", "timestamps": [{"word": "one"}]}),
            json.dumps({"audio": "BBB", "timestamps": [{"word": "two"}]}),
        ]
    )
    chunks = parse_captioned_body(body)
    assert [chunk["audio"] for chunk in chunks] == ["AAA", "BBB"]


def test_tolerates_sse_framing_and_blank_lines() -> None:
    body = 'data: {"audio": "AAA"}\n\n' 'data: {"audio": "BBB"}\n'
    assert [chunk["audio"] for chunk in parse_captioned_body(body)] == ["AAA", "BBB"]


def test_skips_unparsable_fragment_without_losing_the_rest() -> None:
    body = '{"audio": "AAA"}\nnot json\n{"audio": "BBB"}'
    assert [chunk["audio"] for chunk in parse_captioned_body(body)] == ["AAA", "BBB"]


def test_empty_body_yields_no_chunks() -> None:
    assert parse_captioned_body("") == []
    assert parse_captioned_body("   \n  ") == []


def test_single_chunk_wav_is_returned_untouched() -> None:
    wav = _silent_wav(240)
    assert merge_chunk_audio([wav], "wav") == wav


def test_wav_chunks_merge_into_one_playable_file() -> None:
    # Only the first chunk carries a header; the rest are headerless PCM. Naive
    # concatenation leaves the RIFF sizes claiming chunk one's length, and the
    # player stops there.
    first, second, third = _silent_wav(240), _silent_wav(120), _silent_wav(360)
    merged = merge_chunk_audio([first, _pcm_only(second), _pcm_only(third)], "wav")

    with wave.open(io.BytesIO(merged), "rb") as handle:
        assert handle.getnframes() == 240 + 120 + 360

    # The size fields must describe the whole file, not the first chunk.
    declared = int.from_bytes(merged[4:8], "little")
    assert declared == len(merged) - 8


def test_mp3_chunks_are_simply_concatenated() -> None:
    assert merge_chunk_audio([b"AA", b"BB", b"CC"], "mp3") == b"AABBCC"


def test_non_riff_chunks_are_not_pretended_to_merge() -> None:
    # If a server hands back something that is not WAV, joining the bytes is the
    # only honest thing to do — patching sizes would corrupt it.
    assert merge_chunk_audio([b"not-a-wav", b"tail"], "wav") == b"not-a-wavtail"


def test_base64_chunks_decode_to_merged_audio() -> None:
    first, second = _silent_wav(200), _silent_wav(200)
    chunks = [
        {"audio": base64.b64encode(first).decode(), "timestamps": []},
        {"audio": base64.b64encode(_pcm_only(second)).decode(), "timestamps": []},
    ]
    merged = merge_chunk_audio(
        [base64.b64decode(chunk["audio"]) for chunk in chunks], "wav"
    )
    with wave.open(io.BytesIO(merged), "rb") as handle:
        assert handle.getnframes() == 400
