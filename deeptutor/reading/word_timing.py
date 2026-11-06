"""Word-timed speech for Immersive Reading (NaturalReader-style highlighting).

The plain ``read-aloud`` endpoint returns audio with no timing metadata, so the
reader can only highlight a whole paragraph. This module adds per-word timing on
top of the existing voice stack:

* Edge voices expose ``WordBoundary`` events (offset/duration in 100-ns ticks).
* Providers that cannot report boundaries still return audio; the client then
  falls back to a proportional estimate.

Everything here is additive: when a provider has no timing support the response
is still valid audio, and the caller keeps working.
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass
from typing import Any

from deeptutor.services.voice import VoiceProviderError, synthesize_speech

logger = logging.getLogger(__name__)

# Word boundaries are timed in 100-nanosecond ticks (Windows file time units).
TICKS_PER_SECOND = 10_000_000

# A horizontal rule / code fence / markdown table pipe is not speakable text.
_UNSPEAKABLE = re.compile(r"^[\s\-_=*#>`|~\[\](){}]+$")


@dataclass(slots=True)
class TimedWord:
    """One spoken word and the moment it starts."""

    word: str
    start: float
    duration: float

    def as_dict(self) -> dict[str, Any]:
        return {"word": self.word, "start": round(self.start, 3), "duration": round(self.duration, 3)}


def _ticks_to_seconds(ticks: Any) -> float:
    try:
        return float(ticks) / TICKS_PER_SECOND
    except (TypeError, ValueError):
        return 0.0


_NUMBER_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
    19: "nineteen", 20: "twenty", 30: "thirty", 40: "forty",
    50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
}


def _number_words(value: str) -> list[str]:
    """The words a voice is likely to say for a numeral in the source text.

    Kokoro reads "2.0" as "two" and "180" as "one hundred eighty". Without this,
    a numeral never matches the spoken token, the forward match desynchronises,
    and every later word falls back to a single timestamp — the highlight then
    sits still for the rest of the passage.
    """
    value = value.replace(",", "")
    if not value.isdigit():
        return []
    number = int(value)
    if number in _NUMBER_WORDS:
        return [_NUMBER_WORDS[number]]
    if number < 100:
        tens, ones = divmod(number, 10)
        words = [_NUMBER_WORDS[tens * 10]] if tens * 10 in _NUMBER_WORDS else []
        if ones and ones in _NUMBER_WORDS:
            words.append(_NUMBER_WORDS[ones])
        return words
    if number < 1000:
        hundreds, rest = divmod(number, 100)
        words = [_NUMBER_WORDS[hundreds], "hundred"]
        if rest:
            words.extend(_number_words(str(rest)))
        return words
    return []


# Typographic quotes and dashes that appear in real prose but never in a voice's
# token stream. `\w` counts U+2019 as a word character in Python, so "freelancer's"
# normalised to "freelancers" while the voice said "freelancer's" — the match
# failed, the cursor stopped, and every later word lost its timing.
_QUOTE_FOLD = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2032": "'",
    "\u00b4": "'", "\u0060": "'",
})


def _normalize_word(token: str) -> str:
    """The spelling a word and its spoken counterpart should agree on."""
    return re.sub(r"[^\w']+", "", token.translate(_QUOTE_FOLD)).casefold()


def _candidate_forms(token: str) -> list[str]:
    """Normalised spellings a token may appear as in the spoken token stream."""
    # Inspect the decimal *before* stripping punctuation: the plain normaliser
    # collapses "2.0" to "20", which would then be read as "twenty".
    trimmed = token.strip()
    decimal = re.match(r"^(\d+)\.(\d+)$", trimmed)
    base = _normalize_word(token)
    if not base:
        return []
    forms = [base]
    head = decimal
    if head and head.group(2).strip("0") == "":
        # A version-style decimal: Kokoro reads "2.0" as "two", not "twenty".
        # Only whole numbers below the decimal point are voiced, so offer the
        # leading digits as the candidate spelling.
        forms.append(_number_words(head.group(1))[0] if _number_words(head.group(1)) else "")
    else:
        plain = re.match(r"^(\d+)$", base)
        if plain:
            whole = _number_words(plain.group(1))
            if whole:
                forms.append(" ".join(whole))
    return [form for form in forms if form]


def align_words_to_text(
    text: str, boundary_times: list[tuple[str, float, float]]
) -> list[TimedWord]:
    """Project TTS boundary strings onto the reader's own word tokens.

    The speech engine may normalise or split words differently from the source
    text ("$12" -> "twelve dollars", "e.g." -> "e g"). Highlighting needs one
    entry per *visible* token, in reading order, so boundaries are matched
    forward against the source tokens instead of being used verbatim.

    A boundary that cannot be matched is skipped rather than shifting every
    later word; unmatched source tokens inherit the current playback position.
    """
    tokens = [
        (match.start(), match.group(0))
        for match in re.finditer(r"[^\s]+", text)
        if match.group(0) and not _UNSPEAKABLE.match(match.group(0))
    ]
    if not tokens:
        return []

    out: list[TimedWord] = []
    cursor = 0
    last_start = 0.0
    # Spoken tokens still owed to a numeral that expanded into several words.
    pending_skip = [0]

    for raw_word, start, duration in boundary_times:
        if pending_skip[0] > 0:
            # This boundary belongs to a numeral already accounted for.
            pending_skip[0] -= 1
            continue
        # Engines can report a first word slightly before zero (Kokoro returned
        # -0.0028s, an artifact of its internal alignment). The API schema
        # rejects negative starts, and because `last_start` is carried forward
        # one negative value poisoned every preceding word too — the endpoint
        # answered 500 and the reader got nothing at all.
        start = max(0.0, start)
        duration = max(0.0, duration)
        wanted = _normalize_word(raw_word)
        if not wanted:
            continue
        matched_at = -1
        # How many spoken tokens this page token accounts for. A numeral expands
        # to several words ("180" -> "one hundred eighty"), and the extra words
        # have no page token of their own; consuming them keeps the cursor in
        # step instead of letting them fall through as unmatched.
        consumed = 0
        for index in range(cursor, len(tokens)):
            forms = _candidate_forms(tokens[index][1])
            if not forms:
                continue
            # Exact normalised match, or a numeral read out ("2.0" -> "two").
            if wanted in forms:
                matched_at = index
                consumed = 1
                break
            expanded = next((form for form in forms if " " in form), None)
            if expanded and wanted == expanded.split()[0]:
                matched_at = index
                consumed = len(expanded.split())
                break
            # Partial match: a boundary word may span tokens ("24/7" -> 24, 7).
            if any(wanted.startswith(form) and form for form in forms):
                matched_at = index
                consumed = 1
                break
        if matched_at < 0:
            continue
        for index in range(cursor, matched_at):
            out.append(TimedWord(tokens[index][1], last_start, max(0.0, start - last_start)))
        out.append(TimedWord(tokens[matched_at][1], start, duration))
        last_start = start
        cursor = matched_at + 1
        if consumed > 1:
            pending_skip[0] += consumed - 1

    # Trailing tokens the engine never reported: hold them at the last position.
    for index in range(cursor, len(tokens)):
        out.append(TimedWord(tokens[index][1], last_start, 0.0))

    # Sort defensively so the client can walk the list monotonically.
    out.sort(key=lambda item: item.start)
    return out


async def synthesize_with_word_timings(
    text: str,
    *,
    voice: str | None = None,
    speed: float | None = None,
    strip_markdown: bool = True,
) -> dict[str, Any]:
    """Return base64 audio plus per-word timings for ``text``.

    ``strip_markdown`` must stay on for text read out of a reading unit, which is
    Markdown: a voice pronounces ``###`` literally, and the markers would be
    counted as words.

    The reader overrides it to ``False``, and that is load-bearing. What it sends
    is the *rendered* article text, which already has no Markdown — but a list
    item on screen still begins with its marker (``1.``), and the page's word
    list includes it. Stripping that text removed the marker from the audio while
    the page still had a slot for it, so the highlight ran one word ahead of the
    voice from the first list item onward. The client's text is the coordinate
    system the highlight indexes against; it must not be rewritten here.

    Raises ``VoiceProviderError`` when the configured provider cannot synthesise
    the text at all; a provider that simply has no boundary events is *not* an
    error, it just yields ``words == []``.
    """
    boundaries: list[tuple[str, float, float]] = []

    # Speak prose, not markup. Reading units carry Markdown (``###``, ``**``,
    # bullet markers), and a voice that does not skip it pronounces the markers:
    # Kokoro says "number number number" for a heading, and the alignment pass
    # counts them as words so every later highlight is offset. The timings are
    # aligned against the *stripped* text for the same reason — they have to
    # describe the audio that was actually produced.
    from deeptutor.services.voice.base import strip_markdown_for_speech

    text = (
        strip_markdown_for_speech(text, math_speak=True).strip()
        if strip_markdown
        else text.strip()
    )
    # A unit that is nothing but markup has nothing to say. `strip_markdown_for_
    # speech` leaves a bare `---` (horizontal rule) behind, so check for leftover
    # unspeakable tokens rather than only testing for an empty string.
    if not text or all(_UNSPEAKABLE.match(token) for token in text.split()):
        return {"audio_base64": "", "mime": "audio/mpeg", "words": [], "timed": False}

    # Edge is the only built-in adapter that reports WordBoundary events, and it
    # needs ``boundary="WordBoundary"`` to emit them at all.
    from deeptutor.services.config.provider_runtime import resolve_tts_runtime_config

    config = resolve_tts_runtime_config()
    provider = (config.provider_name or "").lower()
    if provider in {"", "edge", "edge_tts", "microsoft_edge"}:
        from deeptutor.services.voice.adapters.edge import EDGE_DEFAULT_VOICE, edge_rate

        try:
            import edge_tts
        except ImportError:
            edge_tts = None
        if edge_tts is not None:
            communicate = edge_tts.Communicate(
                text,
                voice or config.voice or EDGE_DEFAULT_VOICE,
                rate=edge_rate(speed if speed is not None else config.speed),
                boundary="WordBoundary",
            )
            audio = bytearray()
            try:
                async for chunk in communicate.stream():
                    kind = chunk.get("type")
                    if kind == "audio":
                        audio += chunk["data"]
                    elif kind == "WordBoundary":
                        boundaries.append(
                            (
                                str(chunk.get("text", "")),
                                _ticks_to_seconds(chunk.get("offset")),
                                _ticks_to_seconds(chunk.get("duration")),
                            )
                        )
            except Exception as exc:  # edge-tts raises its own and aiohttp errors
                raise VoiceProviderError(
                    f"Edge TTS request failed: {type(exc).__name__}."
                ) from exc
            if audio:
                words = align_words_to_text(text, boundaries)
                return {
                    "audio_base64": base64.b64encode(bytes(audio)).decode("ascii"),
                    "mime": "audio/mpeg",
                    "words": [word.as_dict() for word in words],
                    "timed": bool(words),
                }

    # Kokoro reports exact word boundaries through its captioned endpoint, so it
    # gets a real timeline rather than the proportional estimate other providers
    # fall back to.
    if provider == "kokoro":
        from deeptutor.services.voice.adapters.kokoro import synthesize_kokoro_with_timings

        audio_bytes, kokoro_type, kokoro_words = await synthesize_kokoro_with_timings(
            text, config
        )
        # Kokoro's timestamps are the *spoken* tokens, which are not the page's
        # tokens: it splits a list marker into a word and a punctuation mark
        # ("1." -> "one" + "."), so a unit with three list items yields three
        # more timings than the page has words. Handing those straight to the
        # reader made the highlight index into a longer list than the page and
        # run ahead of the voice. Project them onto the page's own tokens, the
        # same way the Edge path does.
        boundaries = [
            (str(word.get("word", "")), float(word.get("start", 0.0)), float(word.get("duration", 0.0)))
            for word in kokoro_words
        ]
        words = align_words_to_text(text, boundaries)
        return {
            "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
            "mime": kokoro_type,
            "words": [word.as_dict() for word in words],
            "timed": bool(words),
        }

    # Any other configured provider: plain audio, no timings.
    audio, content_type = await synthesize_speech(text)
    from deeptutor.services.voice.audio import _parse_pcm_content_type, _pcm16_to_wav

    pcm = _parse_pcm_content_type(content_type)
    if pcm:
        audio = _pcm16_to_wav(audio, sample_rate=pcm[0], channels=pcm[1])
        content_type = "audio/wav"
    return {
        "audio_base64": base64.b64encode(audio).decode("ascii"),
        "mime": content_type,
        "words": [],
        "timed": False,
    }


__all__ = [
    "TICKS_PER_SECOND",
    "TimedWord",
    "align_words_to_text",
    "synthesize_with_word_timings",
]
