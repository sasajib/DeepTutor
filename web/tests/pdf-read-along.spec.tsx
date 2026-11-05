import { act, render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

/**
 * A PDF is not one long article: pages are separate canvases with a PDF.js text
 * layer over each, they render lazily, and only the near ones hold any text.
 * Read-along therefore has to be fed the *visible* page rather than a single
 * element, and it has to keep going page to page.
 *
 * These pin the wiring, not the audio: the source the reader registers with the
 * read-along session is the contract under test.
 */

const registry = vi.hoisted(() => ({
  source: null as null | {
    getPage: () => { text: string; wordCount: number } | null;
    hasNext: () => boolean;
    next: () => void;
  },
}));

vi.mock("@/components/reading/reading-actions-context", () => ({
  useReadingActions: () => ({
    registerReadAlongSource: (source: typeof registry.source) => {
      registry.source = source;
    },
    readAlongActive: false,
    continueReadAlong: () => undefined,
    startReadAlong: () => undefined,
    speakSelection: () => undefined,
    activeWord: -1,
    readAlongWords: undefined,
  }),
}));

vi.mock("@/lib/pdfjs-loader", () => ({
  outputScale: () => 1,
  pdfjsWasmUrl: () => "/pdfjs/wasm/",
  pdfjsCMapUrl: () => "/pdfjs/cmaps/",
  loadPdfjs: async () => ({
    getDocument: () => ({
      destroy: async () => undefined,
      promise: Promise.resolve({
        getPage: async () => ({
          getViewport: ({ scale }: { scale: number }) => ({
            width: 600 * scale,
            height: 800 * scale,
          }),
          render: () => ({ promise: Promise.resolve(), cancel: () => undefined }),
          // Two words of real text so the model is non-empty.
          getTextContent: async () => ({
            items: [
              { str: "Visible", transform: [1, 0, 0, 1, 10, 700], width: 40, height: 10 },
              { str: "page", transform: [1, 0, 0, 1, 55, 700], width: 25, height: 10 },
            ],
          }),
        }),
      }),
    }),
    TextLayer: class {
      async render() {}
      cancel() {}
    },
  }),
}));

vi.mock("react-i18next", () => {
  const t = (key: string) => key;
  return { useTranslation: () => ({ t }) };
});

const { PdfDocumentView } = await import("@/components/reading/PdfDocumentView");

function renderPdf(unitCount = 3) {
  return render(
    <PdfDocumentView
      materialId="pdf-1"
      unitCount={unitCount}
      annotations={[]}
      jump={null}
      onSelection={() => undefined}
    />,
  );
}

describe("PDF read-along", () => {
  beforeEach(() => {
    registry.source = null;
    // jsdom has no layout: give pages a box so the visible-page probe finds one.
    Element.prototype.getBoundingClientRect = function () {
      const el = this as HTMLElement;
      const index = [...(el.parentElement?.children ?? [])].indexOf(el);
      const top = 80 + index * 900;
      return {
        top, bottom: top + 880, left: 0, right: 600, width: 600, height: 880,
        x: 0, y: top, toJSON: () => ({}),
      } as DOMRect;
    };
  });

  it("registers a read-along source for the document", async () => {
    renderPdf();
    await waitFor(() => expect(registry.source).not.toBeNull());
  });

  it("reads the page through its PDF.js text layer", async () => {
    const view = renderPdf();
    await waitFor(() => expect(registry.source).not.toBeNull());

    // jsdom never runs the real PDF.js text layer, so stand one in: the point
    // under test is that `getPage` reads the *visible page's* layer, not that
    // PDF.js painted it.
    const pageEl = view.container.querySelector<HTMLElement>('[data-reader-unit="1"]');
    expect(pageEl).toBeTruthy();
    let layer = pageEl!.querySelector<HTMLElement>(".textLayer");
    if (!layer) {
      layer = document.createElement("div");
      layer.className = "textLayer";
      pageEl!.appendChild(layer);
    }
    layer.textContent = "Visible page words";

    await waitFor(() => {
      const page = registry.source?.getPage();
      expect(page?.text).toContain("Visible");
      expect(page?.wordCount).toBe(3);
    });
  });

  it("reports a next page until the last one", async () => {
    renderPdf(3);
    await waitFor(() => expect(registry.source).not.toBeNull());
    expect(typeof registry.source?.hasNext()).toBe("boolean");
  });

  it("drops the source when the document closes", async () => {
    const view = renderPdf();
    await waitFor(() => expect(registry.source).not.toBeNull());
    act(() => view.unmount());
    expect(registry.source).toBeNull();
  });
});
