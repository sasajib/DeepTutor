"use client";

import { useEffect, useRef } from "react";

import { useReadingActions } from "@/components/reading/reading-actions-context";
import {
  collectSpeakableText,
  rangeForWordIndex,
} from "@/lib/reading-speakable-text";

const HIGHLIGHT_NAME = "reading-readalong";

/**
 * Read-along highlight: tints the word currently being spoken.
 *
 * Uses the **CSS Custom Highlight API** rather than wrapping words in spans.
 * That matters here: the reader anchors annotations to quotes and character
 * offsets inside the article, so adding thousands of `span` nodes mid-passage
 * would invalidate stored anchors and fight React over the DOM it owns.
 * `CSS.highlights` paints a `Range` without mutating the document at all, and
 * the whole effect disappears with a single `delete`.
 *
 * The timeline comes from `useReadAloudSpeech`: real server word boundaries when
 * the voice reports them, otherwise a duration-based estimate.
 */
export function ReadAlongHighlighter({
  containerRef,
}: {
  containerRef: React.RefObject<HTMLElement | null>;
}) {
  const shared = useReadingActions();
  const activeWord = shared?.activeWord ?? -1;
  const words = shared?.readAlongWords;
  // The word being spoken, kept so the header can take the reader to it on
  // demand — the cursor is not always on the page they are looking at.
  const rangeRef = useRef<Range | null>(null);

  useEffect(() => {
    const registry = getHighlightRegistry();
    const root = containerRef.current;
    if (!registry || !root || activeWord < 0 || !words?.length) {
      registry?.delete(HIGHLIGHT_NAME);
      return;
    }

    const HighlightCtor = getHighlightCtor();
    // Same enumeration the click handler used to pick the start word, so
    // index N means the same word in both directions.
    const range = rangeForWordIndex(collectSpeakableText(root), activeWord);
    if (!HighlightCtor || !range) {
      registry.delete(HIGHLIGHT_NAME);
      return;
    }

    registry.set(HIGHLIGHT_NAME, new HighlightCtor(range));
    rangeRef.current = range;
    keepWordInView(range);

    return () => {
      registry.delete(HIGHLIGHT_NAME);
    };
  }, [activeWord, containerRef, words]);

  // Let the reader's toolbar bring them back to the spoken word. Read-along can
  // be well ahead of — or behind — the page on screen, so "where am I?" needs an
  // answer that does not depend on the cursor happening to be visible.
  const register = shared?.registerReadAlongReveal;
  useEffect(() => {
    if (!register) return;
    const cleanup = register(() => revealWord(rangeRef.current));
    return typeof cleanup === "function" ? cleanup : () => register(null);
  }, [register]);

  // Never leave a stale highlight behind when the reader closes or speech stops.
  useEffect(() => {
    const registry = getHighlightRegistry();
    return () => {
      registry?.delete(HIGHLIGHT_NAME);
    };
  }, []);

  return null;
}

/** The element a range's opening mark sits in. */
function startElement(range: Range): HTMLElement | null {
  const container = range.startContainer;
  if (container.nodeType === Node.ELEMENT_NODE) return container as HTMLElement;
  return container.parentElement;
}

/** The box that actually scrolls for this document, if any. */
function nearestScroller(from: HTMLElement | null): HTMLElement | null {
  let node: HTMLElement | null = from;
  while (node) {
    const overflow = getComputedStyle(node).overflowY;
    if (/(auto|scroll)/.test(overflow) && node.scrollHeight > node.clientHeight) {
      return node;
    }
    node = node.parentElement;
  }
  return null;
}

/**
 * Scroll ``scroller`` by the smallest amount that moves ``rect`` into view.
 *
 * `scrollIntoView` on the word was wrong: it walks every ancestor and, when the
 * word belonged to a page far down the document, threw the reader thirty
 * thousand pixels away from where they were reading. Adjusting one known
 * scroller by a measured delta keeps the movement local and bounded.
 */
function scrollRangeIntoView(
  range: Range | null,
  { center = false }: { center?: boolean } = {},
): void {
  if (!range) return;
  const start = startElement(range);
  if (!start) return;

  let rect: DOMRect;
  try {
    rect = range.getBoundingClientRect();
  } catch {
    return;
  }
  if (!rect || (rect.width === 0 && rect.height === 0)) return;

  const scroller = nearestScroller(start);
  const margin = 48;
  let delta: number;

  if (scroller) {
    const box = scroller.getBoundingClientRect();
    const top = rect.top - box.top;
    const height = scroller.clientHeight;
    if (center) {
      delta = top - (height - rect.height) / 2;
    } else if (top < margin) {
      delta = top - margin;
    } else if (top + rect.height > height - margin) {
      delta = top + rect.height - (height - margin);
    } else {
      return;
    }
    if (Math.abs(delta) < 1) return;
    // Instant, not smooth: this reader has to work even where smooth scrolling is
    // suppressed (reduced-motion settings, some embedded engines), and there a
    // smooth request is silently dropped — the word stays off screen.
    scroller.scrollTo?.({ top: scroller.scrollTop + delta, behavior: "auto" });
    return;
  }

  const viewport = window.innerHeight;
  if (center) {
    delta = rect.top - (viewport - rect.height) / 2;
  } else if (rect.top < margin) {
    delta = rect.top - margin;
  } else if (rect.bottom > viewport - margin) {
    delta = rect.bottom - (viewport - margin);
  } else {
    return;
  }
  if (Math.abs(delta) < 1) return;
  window.scrollTo?.({ top: window.scrollY + delta, behavior: "auto" });
}

/**
 * Follow the voice only while the reader is already on the spoken page.
 *
 * A page of prose is taller than the viewport, so without this the highlight
 * runs below the fold. But when the cursor is on some other page entirely —
 * read-along was started elsewhere, or the reader scrolled away — yanking them
 * across the document is worse than leaving them be: the toolbar's locate button
 * is how they ask to be taken there.
 */
function keepWordInView(range: Range): void {
  const start = startElement(range);
  if (!start) return;

  let rect: DOMRect;
  try {
    rect = range.getBoundingClientRect();
  } catch {
    return;
  }
  if (!rect || (rect.width === 0 && rect.height === 0)) return;

  const scroller = nearestScroller(start);
  const height = scroller?.clientHeight ?? window.innerHeight;
  const box = scroller?.getBoundingClientRect();
  const distance = box ? Math.abs(rect.top - box.top) : Math.abs(rect.top);
  // More than a screen and a half away: a different page, not the next line.
  if (distance > height * 1.5) return;

  scrollRangeIntoView(range);
}

/** Take the reader to the spoken word, wherever in the document it is. */
function revealWord(range: Range | null): void {
  scrollRangeIntoView(range, { center: true });
}

type HighlightCtor = new (...ranges: Range[]) => unknown;

interface HighlightRegistry {
  set: (name: string, highlight: unknown) => void;
  delete: (name: string) => void;
}

function getHighlightCtor(): HighlightCtor | null {
  if (typeof window === "undefined") return null;
  const ctor = (window as unknown as { Highlight?: HighlightCtor }).Highlight;
  return typeof ctor === "function" ? ctor : null;
}

function getHighlightRegistry(): HighlightRegistry | null {
  if (typeof CSS === "undefined") return null;
  const registry = (CSS as unknown as { highlights?: Map<string, unknown> }).highlights;
  if (!registry || typeof registry.set !== "function") return null;
  return registry as unknown as HighlightRegistry;
}
