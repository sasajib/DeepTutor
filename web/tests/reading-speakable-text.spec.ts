import { describe, expect, it } from "vitest";

import {
  collectSpeakableText,
  rangeForWordIndex,
  speakableFrom,
  wordIndexAtPoint,
  wordIndexForRangeStart,
} from "@/lib/reading-speakable-text";

function article(html: string): HTMLElement {
  const root = document.createElement("article");
  root.innerHTML = html;
  document.body.append(root);
  return root;
}

describe("collectSpeakableText", () => {
  it("enumerates words in document order and joins them for the voice", () => {
    const root = article("<p>Alpha bravo</p><p>charlie delta</p>");
    const model = collectSpeakableText(root);

    expect(model.words.map((w) => w.word)).toEqual([
      "Alpha",
      "bravo",
      "charlie",
      "delta",
    ]);
    // The voice gets plain words, not the markup.
    expect(model.text).toBe("Alpha bravo charlie delta");
  });

  it("strips inline markup down to the words on screen", () => {
    const root = article("<p>The <strong>quick</strong> brown <em>fox</em></p>");
    const model = collectSpeakableText(root);

    expect(model.text).toBe("The quick brown fox");
  });

  it("skips code, which the voice does not read as prose", () => {
    const root = article(
      "<p>Run this</p><pre><code>npm install foo</code></pre><p>then stop</p>",
    );
    const model = collectSpeakableText(root);

    expect(model.words.map((w) => w.word)).toEqual([
      "Run",
      "this",
      "then",
      "stop",
    ]);
  });

  it("keeps punctuation attached so the reader can still be matched", () => {
    const root = article("<p>It costs $12 today.</p>");
    expect(collectSpeakableText(root).words.map((w) => w.word)).toEqual([
      "It",
      "costs",
      "$12",
      "today.",
    ]);
  });

  it("ignores whitespace-only nodes", () => {
    const root = article("<p>one</p>\n   \n<p>two</p>");
    expect(collectSpeakableText(root).text).toBe("one two");
  });
});

describe("wordIndexAtPoint", () => {
  it("finds the word under a caret inside a text node", () => {
    const root = article("<p>Alpha bravo charlie</p>");
    const model = collectSpeakableText(root);
    const node = model.words[0].node;

    expect(wordIndexAtPoint(model, node, 0)).toBe(0);
    expect(wordIndexAtPoint(model, node, 6)).toBe(1);
    expect(wordIndexAtPoint(model, node, 12)).toBe(2);
  });

  it("advances to the next word when the caret sits in the whitespace", () => {
    const root = article("<p>Alpha bravo</p>");
    const model = collectSpeakableText(root);

    // Offset 5 is the space between "Alpha" and "bravo".
    expect(wordIndexAtPoint(model, model.words[0].node, 5)).toBe(1);
  });
});

describe("wordIndexForRangeStart", () => {
  it("maps a selection start onto the word it begins in", () => {
    const root = article("<p>Alpha bravo charlie</p>");
    const model = collectSpeakableText(root);
    const node = model.words[0].node;
    const range = document.createRange();
    range.setStart(node, 6);
    range.setEnd(node, 11);

    expect(wordIndexForRangeStart(model, range)).toBe(1);
  });
});

describe("rangeForWordIndex", () => {
  it("covers exactly one word and nothing around it", () => {
    const root = article("<p>Alpha bravo charlie</p>");
    const model = collectSpeakableText(root);

    expect(rangeForWordIndex(model, 1)?.toString()).toBe("bravo");
    expect(rangeForWordIndex(model, 99)).toBeNull();
  });
});

describe("speakableFrom", () => {
  it("returns the tail of the page and the offset it started at", () => {
    const root = article("<p>Alpha bravo charlie delta</p>");
    const model = collectSpeakableText(root);

    expect(speakableFrom(model, 2)).toEqual({
      text: "charlie delta",
      base: 2,
    });
  });

  it("speaks the whole page when starting at the top", () => {
    const root = article("<p>Alpha bravo</p>");
    const model = collectSpeakableText(root);
    expect(speakableFrom(model, 0)).toEqual({ text: "Alpha bravo", base: 0 });
  });

  it("clamps a start past the end rather than throwing", () => {
    const root = article("<p>Alpha bravo</p>");
    const model = collectSpeakableText(root);
    expect(speakableFrom(model, 99)).toEqual({ text: "", base: 2 });
  });
});

describe("selection offset on the page (read-along regression)", () => {
  // The bug: a selection three paragraphs down spoke correctly but highlighted
  // near the top of the page, because the selection started its timeline at word
  // zero of the document instead of at the selection's own first word.
  const LONG = `
    <h2>Preface</h2>
    <p>Alpha bravo charlie delta echo foxtrot golf hotel.</p>
    <p>India juliet kilo lima mike november oscar papa.</p>
    <p>Quebec romeo sierra tango uniform victor whiskey.</p>
  `;

  function selectIn(root: HTMLElement, paragraphIndex: number, start: number, end: number) {
    const paragraphs = [...root.querySelectorAll("p")];
    const node = paragraphs[paragraphIndex].firstChild as Text;
    const range = document.createRange();
    range.setStart(node, start);
    range.setEnd(node, end);
    return range;
  }

  it("reports where a mid-page selection starts", () => {
    const root = article(LONG);
    const model = collectSpeakableText(root);

    // Heading (1) + first paragraph (8) + second paragraph (8) = 17 words before
    // the third paragraph.
    const range = selectIn(root, 2, 0, 12); // "Quebec romeo"
    const base = wordIndexForRangeStart(model, range);

    expect(base).toBe(17);
    expect(model.words[base].word).toBe("Quebec");
  });

  it("points the timeline at the selected words, not the document top", () => {
    const root = article(LONG);
    const model = collectSpeakableText(root);
    const range = selectIn(root, 2, 0, 13); // "Quebec romeo "
    const base = wordIndexForRangeStart(model, range);

    // What the voice is given, and what the highlight will address.
    const spoken = speakableFrom(model, base);
    expect(spoken.base).toBe(17);
    expect(spoken.text.startsWith("Quebec romeo")).toBe(true);

    // Word 0 of the spoken passage must resolve to "Quebec", not "Preface".
    expect(rangeForWordIndex(model, spoken.base)?.toString()).toBe("Quebec");
    expect(rangeForWordIndex(model, spoken.base + 1)?.toString()).toBe("romeo");
  });

  it("handles a selection that starts on an element boundary", () => {
    const root = article(LONG);
    const model = collectSpeakableText(root);
    const paragraphs = [...root.querySelectorAll("p")];
    const range = document.createRange();
    // Start at the paragraph element itself (what a double-click or a
    // triple-click sometimes produces), not inside its text node.
    range.setStart(paragraphs[1], 0);
    range.setEnd(paragraphs[1], 1);

    const base = wordIndexForRangeStart(model, range);
    expect(base).toBe(9); // heading + first paragraph
    expect(model.words[base].word).toBe("India");
  });
});
