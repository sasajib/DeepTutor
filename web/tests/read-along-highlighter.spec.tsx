import { render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

/**
 * The highlight is painted with the CSS Custom Highlight API, which jsdom does
 * not implement, so the registry is stubbed and the assertions are on the range
 * handed to it. Two behaviours matter: the right word is tinted, and a word that
 * has scrolled off screen is brought back — otherwise the voice reads on while
 * the reader sees nothing marked.
 */

class FakeHighlight {
  ranges: Range[];
  constructor(...ranges: Range[]) {
    this.ranges = ranges;
  }
}

const highlights = new Map<string, FakeHighlight>();

const state = vi.hoisted(() => ({
  activeWord: -1,
  readAlongWords: [] as { word: string; start: number; duration: number }[],
}));

vi.mock("@/components/reading/reading-actions-context", () => ({
  useReadingActions: () => state,
}));

const { ReadAlongHighlighter } = await import(
  "@/components/reading/ReadAlongHighlighter"
);

/** A prose paragraph with a stubbed layout: top of the page starts at `top`. */
function makeContainer(text: string, top: number): HTMLParagraphElement {
  const element = document.createElement("p");
  element.textContent = text;
  element.scrollIntoView = vi.fn();
  document.body.appendChild(element);
  // jsdom has no layout, so ranges cannot be measured. Report the word as
  // sitting at `top`; the component only uses this to decide whether to scroll.
  const original = Range.prototype.getBoundingClientRect;
  Range.prototype.getBoundingClientRect = () =>
    ({
      top,
      bottom: top + 20,
      left: 0,
      right: 40,
      width: 40,
      height: 20,
      x: 0,
      y: top,
      toJSON: () => ({}),
    }) as DOMRect;
  element.dataset.originalRect = String(Boolean(original));
  return element;
}

const words = (line: string) =>
  line.split(/\s+/).map((word, index) => ({
    word,
    start: index,
    duration: 1,
  }));

describe("read-along highlighter", () => {
  beforeEach(() => {
    highlights.clear();
    state.activeWord = -1;
    state.readAlongWords = [];
    (globalThis as unknown as { Highlight: unknown }).Highlight = FakeHighlight;
    (CSS as unknown as { highlights: unknown }).highlights = {
      set: (name: string, highlight: FakeHighlight) => highlights.set(name, highlight),
      get: (name: string) => highlights.get(name),
      delete: (name: string) => highlights.delete(name),
    };
  });

  it("paints the word at the active index", () => {
    const line = "alpha beta gamma";
    const element = makeContainer(line, 100);
    state.readAlongWords = words(line);
    state.activeWord = 1;

    const ref = { current: element } as React.RefObject<HTMLElement | null>;
    const view = render(<ReadAlongHighlighter containerRef={ref} />);

    expect(highlights.get("reading-readalong")?.ranges).toHaveLength(1);
    view.unmount();
    element.remove();
  });

  it("clears the highlight when no word is active", () => {
    const line = "alpha beta gamma";
    const element = makeContainer(line, 100);
    state.readAlongWords = words(line);
    state.activeWord = -1;

    const ref = { current: element } as React.RefObject<HTMLElement | null>;
    const view = render(<ReadAlongHighlighter containerRef={ref} />);

    expect(highlights.get("reading-readalong")).toBeUndefined();
    view.unmount();
    element.remove();
  });

  it("brings a word below the viewport back into view", () => {
    const line = "alpha beta gamma";
    const element = makeContainer(line, window.innerHeight + 2000);
    state.readAlongWords = words(line);
    state.activeWord = 1;

    const ref = { current: element } as React.RefObject<HTMLElement | null>;
    const view = render(<ReadAlongHighlighter containerRef={ref} />);

    expect(element.scrollIntoView).toHaveBeenCalled();
    // Scroll the least distance that reveals it rather than recentring.
    expect(element.scrollIntoView).toHaveBeenCalledWith(
      expect.objectContaining({ block: "end" }),
    );
    view.unmount();
    element.remove();
  });

  it("leaves the page alone when the word is already visible", () => {
    const line = "alpha beta gamma";
    const element = makeContainer(line, 200);
    state.readAlongWords = words(line);
    state.activeWord = 1;

    const ref = { current: element } as React.RefObject<HTMLElement | null>;
    const view = render(<ReadAlongHighlighter containerRef={ref} />);

    expect(element.scrollIntoView).not.toHaveBeenCalled();
    view.unmount();
    element.remove();
  });
});
