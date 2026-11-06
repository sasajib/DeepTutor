import { render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * The highlight is painted with the CSS Custom Highlight API, which jsdom does
 * not implement, so the registry is stubbed and the assertions are on the range
 * handed to it.
 *
 * The scrolling behaviour is what these cover in depth. `scrollIntoView` on the
 * word walks every ancestor, and when the spoken word belonged to a page far
 * down the document it threw the reader tens of thousands of pixels away from
 * where they were reading — so following is now bounded, and an explicit
 * "go to reading position" action takes them there on request.
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
  reveal: null as (() => void) | null,
}));

vi.mock("@/components/reading/reading-actions-context", () => ({
  useReadingActions: () => ({
    activeWord: state.activeWord,
    readAlongWords: state.readAlongWords,
    registerReadAlongReveal: (reveal: (() => void) | null) => {
      state.reveal = reveal;
    },
  }),
}));

const { ReadAlongHighlighter } = await import(
  "@/components/reading/ReadAlongHighlighter"
);

const previousRect = Range.prototype.getBoundingClientRect;

/** Report the highlighted word as sitting at `top` in a 1000px-tall window. */
function wordAt(top: number) {
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
}

function mount(line: string, activeWord: number) {
  state.readAlongWords = line.split(/\s+/).map((word, index) => ({
    word,
    start: index,
    duration: 1,
  }));
  state.activeWord = activeWord;
  const element = document.createElement("p");
  element.textContent = line;
  document.body.appendChild(element);
  const ref = { current: element } as React.RefObject<HTMLElement | null>;
  const view = render(<ReadAlongHighlighter containerRef={ref} />);
  return { view, element };
}

describe("read-along highlighter", () => {
  const scrollTo = vi.fn();

  beforeEach(() => {
    highlights.clear();
    state.activeWord = -1;
    state.readAlongWords = [];
    state.reveal = null;
    scrollTo.mockClear();
    window.scrollTo = scrollTo as unknown as typeof window.scrollTo;
    (globalThis as unknown as { Highlight: unknown }).Highlight = FakeHighlight;
    Object.defineProperty(window, "innerHeight", {
      value: 1000,
      configurable: true,
      writable: true,
    });
    (CSS as unknown as { highlights: unknown }).highlights = {
      set: (name: string, highlight: FakeHighlight) => highlights.set(name, highlight),
      get: (name: string) => highlights.get(name),
      delete: (name: string) => highlights.delete(name),
    };
  });

  afterEach(() => {
    Range.prototype.getBoundingClientRect = previousRect;
  });

  it("paints the word at the active index", () => {
    wordAt(100);
    const { view, element } = mount("alpha beta gamma", 1);

    expect(highlights.get("reading-readalong")?.ranges).toHaveLength(1);
    view.unmount();
    element.remove();
  });

  it("clears the highlight when no word is active", () => {
    wordAt(100);
    const { view, element } = mount("alpha beta gamma", -1);

    expect(highlights.get("reading-readalong")).toBeUndefined();
    view.unmount();
    element.remove();
  });

  it("brings a word just below the fold back into view", () => {
    wordAt(1400); // one screen down: the reader is on this page
    const { view, element } = mount("alpha beta gamma", 1);

    expect(scrollTo).toHaveBeenCalled();
    view.unmount();
    element.remove();
  });

  it("leaves the reader alone when the word is on a distant page", () => {
    // This is the bug that lost the reader's place: the spoken word was on a
    // page thirty thousand pixels away and following it moved them there.
    wordAt(30_000);
    const { view, element } = mount("alpha beta gamma", 1);

    expect(scrollTo).not.toHaveBeenCalled();
    view.unmount();
    element.remove();
  });

  it("goes to a distant word when asked, even though it does not follow", () => {
    wordAt(30_000);
    const { view, element } = mount("alpha beta gamma", 1);
    expect(scrollTo).not.toHaveBeenCalled();

    state.reveal?.();

    expect(scrollTo).toHaveBeenCalled();
    view.unmount();
    element.remove();
  });
});
