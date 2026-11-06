"""Kokoro text-to-speech, running as a local OpenAI-compatible server.

Port of the Kokoro integration from our own reader, where it was the voice the
read-along highlight was built against. Kokoro-FastAPI serves the usual
``/v1/audio/speech`` (so it works with the generic OpenAI-compatible adapter)
plus ``/dev/captioned_speech``, which returns **exact** word timestamps. Exact
timings are the reason this adapter exists: the reader's highlight can only
advance proportionally when a provider reports no boundaries, and Kokoro would
otherwise be a step down from Edge for read-along.

Three things this adapter has to get right, each found the hard way:

* ``/dev/captioned_speech`` answers with **newline-delimited JSON** — one object
  per internal synthesis chunk. A plain ``json.loads`` sees only the first
  object, so long passages lose everything after their first chunk. Both shapes
  are accepted here.
* Only the first chunk carries a WAV header; the rest are headerless PCM. They
  are joined by keeping the first header and patching the RIFF/data sizes —
  naive concatenation produces a file whose player stops after chunk one.
* Kokoro timestamps punctuation as its own token (``","``), which has no
  counterpart in the reader's word list. The aligner drops those.

The server is CPU-bound and runs near real time, so requests are serialised
behind a small concurrency limit rather than fired in parallel.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from typing import Any

import httpx

from deeptutor.services.voice.base import (
    BaseTTSAdapter,
    VoiceProviderError,
    VoiceProviderHTTPError,
    build_auth_headers,
)
from deeptutor.services.voice.config import TTSConfig

logger = logging.getLogger(__name__)

DEFAULT_KOKORO_VOICE = "af_heart"
DEFAULT_KOKORO_TIMEOUT = 120

# Kokoro on CPU synthesises at roughly real time. Let a couple of requests in
# flight, no more: beyond that the queue just grows and every caller times out.
# One narration at a time. Kokoro is a CPU server, and two captioned requests at
# once under load came back partially streamed: the audio stopped early, the word
# timings covered only its first seconds, and the reader's highlight froze for the
# rest of the page. Queueing costs a little latency and keeps every page correct.
_MAX_CONCURRENCY = 1
_slots: dict[int, asyncio.Semaphore] = {}


def _slot_for(loop: asyncio.AbstractEventLoop) -> asyncio.Semaphore:
    """One semaphore per event loop.

    A module-level ``Semaphore`` binds to the loop that first awaited it, which
    breaks under repeated ``asyncio.run`` calls (tests, CLI entry points).
    """
    key = id(loop)
    slot = _slots.get(key)
    if slot is None:
        slot = asyncio.Semaphore(_MAX_CONCURRENCY)
        _slots[key] = slot
    return slot


def normalize_kokoro_base(raw: str | None) -> str:
    """Return the server root, tolerating a trailing slash or ``/v1``.

    Kokoro-FastAPI mounts the speech endpoints at the root when the OpenAI
    compatibility layer is on, but people paste the base with ``/v1`` because
    that is what every other provider uses. Accept both.
    """
    value = (raw or "").strip().rstrip("/")
    if value.endswith("/v1"):
        value = value[: -len("/v1")]
    return value


def _content_type_for(fmt: str) -> str:
    return {"wav": "audio/wav", "pcm": "audio/pcm", "flac": "audio/flac", "opus": "audio/opus"}.get(
        fmt, "audio/mpeg"
    )


def parse_captioned_body(raw: str) -> list[dict[str, Any]]:
    """Split a captioned-speech response into its per-chunk objects.

    Kokoro-FastAPI streams newline-delimited JSON. Older builds return a single
    JSON object, and some proxies wrap lines in SSE ``data:`` framing; all three
    are handled, and an unparsable fragment is skipped rather than discarding
    the whole response.
    """
    text = (raw or "").strip()
    if not text:
        return []

    try:
        single = json.loads(text)
    except json.JSONDecodeError:
        single = None
    if isinstance(single, dict):
        return [single]

    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if candidate.startswith("data:"):
            candidate = candidate[len("data:") :].strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


def _wav_data_bounds(buf: bytes) -> tuple[int, int] | None:
    """Locate the PCM payload of a RIFF/WAVE buffer as ``(offset, length)``."""
    if len(buf) < 12 or buf[0:4] != b"RIFF" or buf[8:12] != b"WAVE":
        return None
    offset = 12
    while offset + 8 <= len(buf):
        chunk_id = buf[offset : offset + 4]
        size = int.from_bytes(buf[offset + 4 : offset + 8], "little")
        if chunk_id == b"data":
            return offset + 8, min(size, len(buf) - offset - 8)
        offset += 8 + size + (size % 2)
    return None


def merge_chunk_audio(buffers: list[bytes], fmt: str) -> bytes:
    """Join per-chunk audio into one playable file.

    MP3 chunks are self-contained, so concatenation is valid. For WAV only the
    first chunk has a header: keep it, append the remaining payloads, then patch
    the RIFF and data sizes to the new total.
    """
    if len(buffers) == 1:
        return buffers[0]
    if fmt != "wav":
        return b"".join(buffers)

    first = buffers[0]
    bounds = _wav_data_bounds(first)
    if bounds is None:
        # Not actually RIFF; do not pretend it merged.
        return b"".join(buffers)

    header = first[: bounds[0]]
    payloads = [first[bounds[0] : bounds[0] + bounds[1]]]
    for extra in buffers[1:]:
        extra_bounds = _wav_data_bounds(extra)
        payloads.append(
            extra[extra_bounds[0] : extra_bounds[0] + extra_bounds[1]]
            if extra_bounds
            else extra
        )
    pcm = b"".join(payloads)

    out = bytearray(header + pcm)
    out[4:8] = (len(out) - 8).to_bytes(4, "little")
    patched = _wav_data_bounds(bytes(out))
    if patched is not None:
        out[patched[0] - 4 : patched[0]] = len(pcm).to_bytes(4, "little")
    return bytes(out)


def _timestamps_to_words(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten chunk timestamps into ``{word, start, duration}`` entries."""
    words: list[dict[str, Any]] = []
    for chunk in chunks:
        for stamp in chunk.get("timestamps") or []:
            if not isinstance(stamp, dict):
                continue
            try:
                start = float(stamp.get("start_time") or 0.0)
                end = float(stamp.get("end_time") or 0.0)
            except (TypeError, ValueError):
                continue
            word = str(stamp.get("word") or "")
            if not word:
                continue
            words.append(
                {"word": word, "start": start, "duration": max(0.0, end - start)}
            )
    return words


async def synthesize_kokoro_with_timings(
    text: str, config: TTSConfig
) -> tuple[bytes, str, list[dict[str, Any]]]:
    """Synthesise ``text`` and return ``(audio, content_type, words)``.

    ``words`` is exact when the server implements ``/dev/captioned_speech`` and
    empty otherwise — a missing caption endpoint is not an error, the caller
    simply falls back to an estimated timeline.
    """
    base = normalize_kokoro_base(config.base_url)
    if not base:
        raise VoiceProviderError("No endpoint URL configured for Kokoro TTS.")

    fmt = (config.response_format or "mp3").lower()
    if fmt not in {"mp3", "wav", "pcm", "flac", "opus"}:
        fmt = "mp3"
    voice = (config.voice or DEFAULT_KOKORO_VOICE).strip() or DEFAULT_KOKORO_VOICE
    speed = config.speed if config.speed and config.speed > 0 else 1.0
    timeout = float(config.request_timeout or DEFAULT_KOKORO_TIMEOUT)
    headers = {
        "Content-Type": "application/json",
        **build_auth_headers(config.auth_style, config.api_key),
    }

    body = {
        "model": config.model or "kokoro",
        "input": text,
        "voice": voice,
        "response_format": fmt,
        "speed": speed,
    }

    async with _slot_for(asyncio.get_running_loop()):
        async with httpx.AsyncClient(timeout=timeout) as client:
            captioned: httpx.Response | None = None
            try:
                captioned = await client.post(
                    f"{base}/dev/captioned_speech",
                    headers=headers,
                    json={**body, "return_timestamps": True},
                )
            except httpx.HTTPError as exc:
                # Endpoint absent on older builds: fall through to plain speech.
                logger.debug("Kokoro captioned speech unavailable: %s", exc)
                captioned = None

            if captioned is not None and captioned.status_code == 200:
                content_type = captioned.headers.get("content-type", "")
                if any(
                    marker in content_type
                    for marker in ("json", "text/plain", "ndjson")
                ):
                    chunks = [
                        chunk
                        for chunk in parse_captioned_body(captioned.text)
                        if chunk.get("audio")
                    ]
                    if chunks:
                        buffers = [
                            base64.b64decode(str(chunk["audio"])) for chunk in chunks
                        ]
                        return (
                            merge_chunk_audio(buffers, fmt),
                            _content_type_for(fmt),
                            _timestamps_to_words(chunks),
                        )

            try:
                response = await client.post(
                    f"{base}/v1/audio/speech", headers=headers, json=body
                )
            except httpx.HTTPError as exc:
                raise VoiceProviderError(
                    f"Kokoro request failed: {type(exc).__name__}."
                ) from exc

    if response.status_code >= 400:
        raise VoiceProviderHTTPError(
            f"Kokoro returned {response.status_code}.",
            status_code=response.status_code,
            body=response.text[:500],
            public_message="Kokoro could not speak this text. Check the Kokoro server is running.",
        )
    return response.content, _content_type_for(fmt), []


class KokoroTTSAdapter(BaseTTSAdapter):
    """Synthesise speech through a Kokoro-FastAPI server."""

    async def synthesize(self, text: str, config: TTSConfig) -> tuple[bytes, str]:
        audio, content_type, _words = await synthesize_kokoro_with_timings(text, config)
        return audio, content_type


__all__ = [
    "DEFAULT_KOKORO_VOICE",
    "KokoroTTSAdapter",
    "merge_chunk_audio",
    "normalize_kokoro_base",
    "parse_captioned_body",
    "synthesize_kokoro_with_timings",
]
