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


def _words_under_100(number: int) -> list[str]:
    if number in _NUMBER_WORDS:
        return [_NUMBER_WORDS[number]]
    if number < 100:
        tens, ones = divmod(number, 10)
        words = [_NUMBER_WORDS[tens * 10]] if tens * 10 in _NUMBER_WORDS else []
        if ones and ones in _NUMBER_WORDS:
            words.append(_NUMBER_WORDS[ones])
        return words
    return []


def _number_words(value: str) -> list[str]:
    """The words a voice is likely to say for a numeral in the source text.

    Voices do not read digits literally. Kokoro says "870" as "eight hundred and
    seventy" and "58" as "fifty-eight". Without an expansion the numeral never
    matches the spoken token, the forward match desynchronises, and every later
    word falls back to a single timestamp — most of the page then has no timing.
    """
    value = value.replace(",", "").replace("\u00a0", "")
    if not value.isdigit():
        return []
    number = int(value)
    if number == 0:
        return ["zero"]
    if number >= 1_000_000_000:
        return []

    parts: list[str] = []
    remaining = number

    for unit, name in ((1_000_000, "million"), (1_000, "thousand")):
        if remaining >= unit:
            count, remaining = divmod(remaining, unit)
            parts.extend(_number_words(str(count)))
            parts.append(name)

    if remaining >= 100:
        hundreds, remaining = divmod(remaining, 100)
        parts.extend([_NUMBER_WORDS[hundreds], "hundred"])
        # British/Indian English, which Kokoro uses, says "and" before a
        # remainder below a hundred: "eight hundred and seventy".
        if remaining:
            parts.append("and")

    if remaining:
        parts.extend(_words_under_100(remaining))

    return parts


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


_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def _squash(value: str) -> str:
    """Letters and digits only.

    Voices punctuate as they please: Kokoro says "seventy-eight" with a hyphen
    and "freelancer's" with one, while the page has "78" and "freelancer’s".
    Comparing on letters and digits alone makes those the same word, and it also
    lets several spoken words match one page token once concatenated.
    """
    return re.sub(r"[^0-9a-z]+", "", value.translate(_QUOTE_FOLD).casefold())


def _decimal_words(digits: str) -> str:
    """A decimal as a voice says it: "0.05" -> "zero point zero five"."""
    return "point " + " ".join(_DIGIT_WORDS.get(digit, digit) for digit in digits)


def _candidate_forms(token: str) -> set[str]:
    """Every squashed spelling a page token may appear as in the spoken stream.

    Voices do not read the source verbatim, so a token is matched against all the
    plausible ways it could be pronounced rather than against its literal text.
    """
    trimmed = token.strip()
    forms: set[str] = set()

    raw = _squash(trimmed)
    if raw:
        forms.add(raw)

    # "78," and "$12." are the same number as "78" and "$12"; only the sentence
    # punctuation differs, and the voice never says it.
    cleaned = trimmed.strip("\"'\u201c\u201d\u2018\u2019()[]{}").rstrip(".,;:!?")

    number = re.match(r"^(\d[\d,]*)(?:\.(\d+))?$", cleaned)
    if number:
        whole = _number_words(number.group(1))
        fraction = number.group(2)
        if fraction:
            # "1.2" is "one point two", four words in all. The bare leading word
            # is *not* offered as a form: "0.05" heard as "zero" would match one
            # word and leave "point zero five" to derail the next real word.
            forms.add(_squash(" ".join(whole) + " " + _decimal_words(fraction)))
            if fraction.strip("0") == "":
                # Version-style "2.0" is just "two" (or "two point").
                forms.add(_squash(" ".join(whole) + " point"))
                forms.add(_squash(" ".join(whole)))
        elif whole:
            forms.add(_squash(" ".join(whole)))

    # "15%" is read "fifteen percent", "$12" as "twelve dollars".
    percent = re.match(r"^(\d[\d,]*)\s*%$", cleaned)
    if percent:
        whole = _number_words(percent.group(1))
        if whole:
            forms.add(_squash(" ".join(whole) + " percent"))
    money = re.match(r"^\$\s*(\d[\d,]*(?:\.\d+)?)$", cleaned)
    if money:
        for form in list(_candidate_forms(money.group(1))):
            forms.add(form)
            forms.add(form + "dollars")

    return {form for form in forms if form}


# A page token can be spoken as several words: "0.05" is four ("zero point zero
# five"), a long numeral more. Matching looks ahead this far.
_MAX_SPOKEN_PER_TOKEN = 12
# How far to look either side to resynchronise when a token has no counterpart.
_MAX_SKIP = 4


def _spread_unspoken_runs(words: list[TimedWord]) -> list[TimedWord]:
    """Give a run of unspoken words its own moments instead of one frozen instant.

    When the voice skips a stretch — a table column it reads elsewhere, a header
    it never voices — every skipped word falls back to the last known position,
    so the highlight sits still while audio plays on. Spreading those words
    across the gap keeps the cursor moving at roughly reading pace until it
    catches the voice again.
    """
    index = 0
    total = len(words)
    while index < total:
        end = index
        while end + 1 < total and words[end + 1].start == words[index].start:
            end += 1
        if end > index and end + 1 < total:
            start = words[index].start
            following = words[end + 1].start
            span = following - start
            count = end - index + 1
            # Only worth doing when the voice really did leave a gap; adjacent
            # words sharing an instant are punctuation, not a skipped passage.
            if span > 0.25:
                step = span / (count + 1)
                for offset in range(1, count):
                    word = words[index + offset]
                    words[index + offset] = TimedWord(
                        word.word, start + step * offset, word.duration
                    )
        index = end + 1
    return words


def align_words_to_text(
    text: str, boundary_times: list[tuple[str, float, float]]
) -> list[TimedWord]:
    """Project TTS boundary strings onto the reader's own word tokens.

    Engines do not read source text verbatim. "78" comes back as "seventy-eight",
    "0.05" as four words, "e.g." as "e g", "$12" as "twelve dollars". The reader
    needs one timing per *visible* token in reading order, so rather than trusting
    the boundaries this walks them forward and matches each run of spoken words
    against the page token it came from, on letters and digits alone.

    Matching is deliberately forgiving in both directions. An unmatched boundary
    resumes at the next page token instead of shifting every later word, and a
    page token the voice skips inherits the current position rather than
    collapsing the rest of the page onto one timestamp.
    """
    tokens = [
        (match.start(), match.group(0))
        for match in re.finditer(r"[^\s]+", text)
        if match.group(0) and not _UNSPEAKABLE.match(match.group(0))
    ]
    if not tokens:
        return []

    # Engines can report a first word slightly before zero (Kokoro returned
    # -0.0028s, an artifact of its internal alignment). The API schema rejects
    # negative starts, and because the position carries forward one negative
    # value poisoned every preceding word too.
    speech = [
        (_squash(raw), max(0.0, start), max(0.0, duration))
        for raw, start, duration in boundary_times
    ]
    speech = [entry for entry in speech if entry[0]]
    if not speech:
        return [TimedWord(word, 0.0, 0.0) for _, word in tokens]

    forms = [_candidate_forms(word) for _, word in tokens]

    def spoken_run(page_index: int, speech_index: int) -> int:
        """How many spoken words this page token accounts for, longest first."""
        candidates = forms[page_index]
        if not candidates:
            return 0
        joined = ""
        best = 0
        for count in range(1, _MAX_SPOKEN_PER_TOKEN + 1):
            position = speech_index + count - 1
            if position >= len(speech):
                break
            joined += speech[position][0]
            if joined in candidates:
                # A longer run is the more specific reading ("0.05" as four
                # words beats the bare "zero" that also fits).
                best = count
        return best

    out: list[TimedWord] = []
    cursor = 0
    position = 0
    last_start = 0.0

    while cursor < len(tokens):
        if position >= len(speech):
            out.append(TimedWord(tokens[cursor][1], last_start, 0.0))
            cursor += 1
            continue

        consumed = spoken_run(cursor, position)
        if consumed:
            start, duration = speech[position][1], speech[position][2]
            out.append(TimedWord(tokens[cursor][1], start, duration))
            last_start = start
            cursor += 1
            position += consumed
            continue

        # No counterpart. Either this page token is never spoken (a stray page
        # number, a table rule) or the voice said something extra. Look a short
        # way in each direction and resume wherever the two agree again.
        resynced = False
        for skip in range(1, _MAX_SKIP + 1):
            if cursor + skip < len(tokens) and spoken_run(cursor + skip, position):
                for index in range(cursor, cursor + skip):
                    out.append(TimedWord(tokens[index][1], last_start, 0.0))
                cursor += skip
                resynced = True
                break
        if resynced:
            continue
        for skip in range(1, _MAX_SKIP + 1):
            if position + skip < len(speech) and spoken_run(cursor, position + skip):
                position += skip
                resynced = True
                break
        if resynced:
            continue

        # Nothing agrees nearby: hold this word at the last known position.
        out.append(TimedWord(tokens[cursor][1], last_start, 0.0))
        cursor += 1
        position += 1

    out.sort(key=lambda item: item.start)
    return _spread_unspoken_runs(out)


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
