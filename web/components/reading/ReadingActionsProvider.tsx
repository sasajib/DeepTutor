"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { fetchAuthStatus } from "@/lib/auth";
import { getOwnLearnerProfile } from "@/lib/profile-api";
import {
  primaryReadingActionRank,
  resolveReadingAgeMode,
  type ReadingAgeMode,
} from "@/lib/reading-age-presentation";
import {
  listReadingExtensions,
  runReadingExtension,
  type ReadingExtensionManifest,
} from "@/lib/reading-api";
import { CHAT_ROUTED_READING_ACTIONS } from "@/lib/reading-passage-prompts";
import { builtInActionLabel } from "./ReadingExtensionBar";
import { useReadAloudSpeech } from "./use-read-aloud-speech";
import {
  ReadingActionsContext,
  type ReadAlongSource,
  type ReadingActionCard,
  type ReadingActionEntry,
  type ReadingActionsValue,
} from "./reading-actions-context";

const MAX_CARDS = 6;

/**
 * Reading actions for a whole workspace: one catalog, one place results land.
 *
 * The reader's selection popover starts them, the companion column shows
 * them, the header's read-aloud button speaks through them. Before this they
 * were a strip of chips above the page that owned its own results, and a
 * learner had two different places an AI answer could appear depending on
 * which button they had pressed.
 */
export function ReadingActionsProvider({
  materialId,
  locator,
  onStart,
  children,
}: {
  /** The material open in the reader; actions run against it. */
  materialId: string | null;
  /** The viewport locator, used when an action has no selection of its own. */
  locator: number;
  /** A result is on its way — the workspace opens the companion for it. */
  onStart?: () => void;
  children: React.ReactNode;
}) {
  const { i18n, t } = useTranslation();
  const [extensions, setExtensions] = useState<ReadingExtensionManifest[]>([]);
  const [ageMode, setAgeMode] = useState<ReadingAgeMode>("default");
  const [cards, setCards] = useState<ReadingActionCard[]>([]);
  const [busyKey, setBusyKey] = useState("");
  const {
    speak,
    speaking,
    stop: stopSpeaking,
    activeWord,
    timeline: readAlongWords,
  } = useReadAloudSpeech();
  const onStartRef = useRef(onStart);
  onStartRef.current = onStart;

  useEffect(() => {
    let active = true;
    void listReadingExtensions()
      .then((rows) => {
        if (active) setExtensions(rows);
      })
      .catch(() => {
        // No catalog means no actions; reading and chat still work.
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    void fetchAuthStatus().then(async (status) => {
      const learnerMode =
        status?.preset === "learner" || Boolean(status?.learning_policy);
      if (!learnerMode) return;
      const profile = await getOwnLearnerProfile().catch(() => null);
      if (active) {
        setAgeMode(
          resolveReadingAgeMode({
            learnerMode,
            profileAge: profile?.age,
            policyAgeBand: status?.learning_policy?.age_band,
          }),
        );
      }
    });
    return () => {
      active = false;
    };
  }, []);

  // A card answers a question about one document; switching documents leaves
  // it describing a page that is no longer on screen.
  useEffect(() => {
    setCards([]);
  }, [materialId]);

  // Leaving the document stops speech. Page turns are *not* in this list:
  // read-along advances units itself, and stopping on every locator change
  // would cancel the chain one step after it started.
  useEffect(() => stopSpeaking, [materialId, stopSpeaking]);

  // -- read-along session --------------------------------------------------
  //
  // NaturalReader's model is a cursor, not a transport: press play once and the
  // voice walks the document, turning pages as it goes; clicking or selecting
  // text anywhere restarts from that point. The session below owns that cursor;
  // the reader only has to say where its words are and how to advance.

  const sourceRef = useRef<ReadAlongSource | null>(null);
  const speakingRef = useRef(false);
  speakingRef.current = speaking;
  // Set when a page finished and another one was requested; cleared when the
  // reader reports the new page. This is what makes continuation single-shot —
  // without it, any re-render could restart the page that just ended.
  const pendingContinueRef = useRef(false);
  // Forward reference so `speakPage`'s watchdog can call the continuation that
  // is declared further down without a circular `useCallback` dependency.
  const continueRef = useRef<(() => void) | null>(null);
  const [readAlongActive, setReadAlongActive] = useState(false);
  // Narrating a page takes the local voice a while; until audio exists there is
  // nothing audible and nothing to highlight, which reads as a dead button. The
  // UI shows this as "preparing" so the wait is visible.
  const [readAlongPreparing, setReadAlongPreparing] = useState(false);
  // How long a page turn may wait for the new page's text to appear.
  const PAGE_WAIT_ATTEMPTS = 40;
  const PAGE_WAIT_MS = 150;
  // Pages in a row that were turned past without being readable. Without a cap,
  // a document whose pages never populate would be walked to the end silently.
  const MAX_SKIPPED_PAGES = 8;
  const pageWaitRef = useRef(0);
  const skippedPagesRef = useRef(0);
  // The word the session should resume from when the reader re-registers after
  // a page turn; -1 means "from the top of the page".
  const resumeAtRef = useRef(0);
  const sessionRef = useRef(0);

  const registerReadAlongSource = useCallback((source: ReadAlongSource | null) => {
    sourceRef.current = source;
  }, []);

  // The highlighter knows where the spoken word sits on screen; the toolbar only
  // needs to ask for it. Several views can be mounted at once (the PDF reader and
  // the plain-text reader), and only the visible one has a laid-out word, so ask
  // them all rather than whichever registered last.
  const revealersRef = useRef<Set<() => void>>(new Set());
  const registerReadAlongReveal = useCallback((reveal: (() => void) | null) => {
    if (!reveal) return;
    revealersRef.current.add(reveal);
    return () => {
      revealersRef.current.delete(reveal);
    };
  }, []);
  const revealReadAlong = useCallback(() => {
    for (const reveal of revealersRef.current) reveal();
  }, []);

  // Speak one page of the document, then hand off to the next one.
  const speakPage = useCallback(
    async (fromWord: number, session: number) => {
      const source = sourceRef.current;
      const page = source?.getPage();
      if (!source || !materialId) return;
      if (!page) {
        // A page turn hands over before the new page has any text to read: the
        // PDF renderer fills a layer only once it is laid out. Waiting briefly
        // keeps the reading going instead of stopping dead on a blank page.
        if (pageWaitRef.current < PAGE_WAIT_ATTEMPTS) {
          pageWaitRef.current += 1;
          window.setTimeout(() => {
            if (sessionRef.current === session) void speakPage(fromWord, session);
          }, PAGE_WAIT_MS);
          return;
        }
        // Still nothing: step past it rather than leaving read-along armed with
        // no voice. A document that has no next page simply finishes.
        pageWaitRef.current = 0;
        skippedPagesRef.current += 1;
        if (source.hasNext() && skippedPagesRef.current <= MAX_SKIPPED_PAGES) {
          resumeAtRef.current = 0;
          source.next();
        } else {
          skippedPagesRef.current = 0;
          setReadAlongActive(false);
        }
        return;
      }
      pageWaitRef.current = 0;
      // `getPage` returns the page's plain text; slice from the requested word.
      const words = page.text.split(/\s+/).filter(Boolean);
      // Starting at or past the last word means "carry on into the next page"
      // rather than re-reading the last word.
      const start = Math.max(0, Math.min(fromWord, words.length));
      const text = words.slice(start).join(" ");
      if (!text) {
        // Nothing left on this page — fall through to the next one.
        if (source.hasNext()) {
          resumeAtRef.current = 0;
          source.next();
        } else {
          setReadAlongActive(false);
        }
        return;
      }
      setReadAlongPreparing(true);
      const played = await speak({
        materialId,
        locator,
        locale: i18n.language,
        fallbackText: text,
        text,
        timed: true,
        base: start,
        onEnd: () => {
          // A stale session (stopped, or restarted elsewhere) must not advance.
          if (sessionRef.current !== session) return;
          if (sourceRef.current?.hasNext()) {
            resumeAtRef.current = 0;
            pendingContinueRef.current = true;
            sourceRef.current.next();
            // Watchdog: the reader normally signals readiness when its new text
            // renders, but a page whose text happens to match the previous one
            // would never re-fire that effect. `continueReadAlong` is
            // single-shot, so a late retry is harmless.
            window.setTimeout(() => {
              if (sessionRef.current === session) continueRef.current?.();
            }, 1200);
          } else {
            // End of the document: read-along finishes on its own.
            setReadAlongActive(false);
          }
        },
      }).finally(() => {
        // Audio exists now (or the attempt failed); the wait is over either way.
        if (sessionRef.current === session) setReadAlongPreparing(false);
      });
      if (!played && sessionRef.current === session) setReadAlongActive(false);
    },
    [i18n.language, locator, materialId, speak],
  );

  // Wait for a turned-to page to have text before speaking it. The renderer fills
  // a page's text layer only once it is laid out, so for a moment after a turn
  // there is nothing to read; giving up there is what left read-along armed and
  // silent with no highlight on screen.
  const beginSpeaking = useCallback(
    async (fromWord: number, session: number) => {
      const source = sourceRef.current;
      if (!source) return;
      if (!source.getPage()) {
        if (pageWaitRef.current < PAGE_WAIT_ATTEMPTS) {
          pageWaitRef.current += 1;
          window.setTimeout(() => {
            if (sessionRef.current === session) void beginSpeaking(fromWord, session);
          }, PAGE_WAIT_MS);
          return;
        }
        pageWaitRef.current = 0;
        skippedPagesRef.current += 1;
        if (source.hasNext() && skippedPagesRef.current <= MAX_SKIPPED_PAGES) {
          resumeAtRef.current = 0;
          source.next();
          // Try again on the page we turned to: the observer only fires when a
          // text layer actually changes, which a blank page never does.
          window.setTimeout(() => {
            if (sessionRef.current === session) void beginSpeaking(0, session);
          }, PAGE_WAIT_MS);
        } else {
          skippedPagesRef.current = 0;
          setReadAlongActive(false);
        }
        return;
      }
      pageWaitRef.current = 0;
      skippedPagesRef.current = 0;
      await speakPage(fromWord, session);
    },
    [speakPage],
  );

  const startReadAlong = useCallback(
    (fromWord?: number) => {
      const source = sourceRef.current;
      if (!source || !materialId) return;
      const session = ++sessionRef.current;
      activeRef.current = true;
      setReadAlongActive(true);
      const target = fromWord ?? resumeAtRef.current;
      resumeAtRef.current = 0;
      void beginSpeaking(target, session);
    },
    [beginSpeaking, materialId],
  );

  // Mirror of `readAlongActive` for callbacks that must not change identity —
  // see `continueReadAlong`.
  const activeRef = useRef(false);
  activeRef.current = readAlongActive;
  const preparingRef = useRef(false);
  preparingRef.current = readAlongPreparing;

  const stopReadAlong = useCallback(() => {
    sessionRef.current += 1;
    activeRef.current = false;
    pendingContinueRef.current = false;
    setReadAlongActive(false);
    setReadAlongPreparing(false);
    stopSpeaking();
  }, [stopSpeaking]);

  const toggleReadAlong = useCallback(() => {
    if (activeRef.current) stopReadAlong();
    else startReadAlong();
  }, [startReadAlong, stopReadAlong]);

  /**
   * Called by the reader once a page's text has rendered and its word model is
   * available. While read-along is armed and nothing is playing — i.e. the
   * previous page just finished — this continues from the top of the new page.
   */
  const continueReadAlongImpl = useCallback(() => {
    // Identity-stable by design: the reader calls this from an effect keyed on
    // its text, and a changing callback would re-run that effect on every
    // render. The `pending` flag makes it single-shot, so a re-render can never
    // restart the page that just finished.
    if (!activeRef.current) return;
    if (!pendingContinueRef.current) return;
    if (speakingRef.current) return;
    pendingContinueRef.current = false;
    startReadAlong(0);
  }, [startReadAlong]);
  continueRef.current = continueReadAlongImpl;
  const continueReadAlong = continueReadAlongImpl;

  /** Speak a selection once, without arming the cursor. */
  const speakSelection = useCallback(
    (text: string, base = 0) => {
      if (!materialId) return;
      const passage = text.trim();
      if (!passage) return;
      sessionRef.current += 1;
      setReadAlongActive(false);
      void speak({
        materialId,
        locator,
        locale: i18n.language,
        fallbackText: passage,
        text: passage,
        timed: true,
        // Index of this selection's first word on the page. The server times the
        // passage from zero, so without this a selection mid-page highlights
        // from the top of the page instead of from where it starts.
        base: Math.max(0, base),
      });
    },
    [i18n.language, locator, materialId, speak],
  );

  const actions = useMemo<ReadingActionEntry[]>(
    () =>
      extensions
        .flatMap((extension) =>
          extension.actions
            .filter(
              (action) =>
                !CHAT_ROUTED_READING_ACTIONS.has(
                  `${extension.id}:${action.id}`,
                ),
            )
            .map((action) => {
              const key = `${extension.id}:${action.id}`;
              const builtIn = builtInActionLabel(extension.id, action.id);
              return {
                key,
                extension,
                action,
                label: builtIn ? t(builtIn) : action.label,
                needsSelection: action.requires.includes("selection"),
              };
            }),
        )
        .sort((left, right) => {
          const leftRank = primaryReadingActionRank(left.key);
          const rightRank = primaryReadingActionRank(right.key);
          return (
            (leftRank < 0 ? 3 : leftRank) - (rightRank < 0 ? 3 : rightRank)
          );
        }),
    [extensions, t],
  );

  const run = useCallback<ReadingActionsValue["run"]>(
    async (entry, target = {}) => {
      if (!materialId) return;
      const selection = (target.selection || "").trim();
      const requestedLocator = target.locator ?? locator;
      const speech = entry.key === "read_aloud:read";
      if (speech && !target.selection) {
        // The toolbar's speaker arms the cursor; speaking is page-by-page.
        toggleReadAlong();
        return;
      }
      const cardId = `${entry.key}-${Date.now()}`;
      const card: ReadingActionCard = {
        id: cardId,
        key: entry.key,
        label: entry.label,
        quote: selection,
        locator: requestedLocator,
        materialId,
        status: "running",
      };
      const open = (row: ReadingActionCard) => {
        onStartRef.current?.();
        setCards((current) => [...current, row].slice(-MAX_CARDS));
      };
      setBusyKey(entry.key);
      // Read-aloud has nothing to show but the header's stop button, so it
      // only gets a card when it fails — silence would read as "it worked".
      if (!speech) open(card);
      const settle = (patch: Partial<ReadingActionCard>) => {
        if (speech) {
          if (patch.status === "error") open({ ...card, ...patch });
          return;
        }
        setCards((current) =>
          current.map((row) => (row.id === cardId ? { ...row, ...patch } : row)),
        );
      };
      try {
        const next = await runReadingExtension(
          materialId,
          entry.extension.id,
          entry.action.id,
          {
            locator: requestedLocator,
            selection,
            locale: i18n.language,
          },
        );
        if (next.type === "browser_speech") {
          const text = String(next.payload.text || "");
          // `timed` asks for per-word boundaries so the page can highlight
          // along with the voice; the hook falls back to an estimate when the
          // active voice cannot report them.
      setReadAlongPreparing(true);
      const played = await speak({
            materialId,
            locator: requestedLocator,
            locale: String(next.payload.locale || i18n.language),
            fallbackText: text,
            timed: true,
          });
          if (!played) {
            settle({
              status: "error",
              error: t("No speech voice is available in this browser."),
            });
          }
          return;
        }
        settle({ status: "done", result: next });
      } catch (error) {
        settle({
          status: "error",
          error: error instanceof Error ? error.message : String(error),
        });
      } finally {
        setBusyKey((current) => (current === entry.key ? "" : current));
      }
    },
    [i18n.language, locator, materialId, speak, t, toggleReadAlong],
  );

  const dismiss = useCallback((cardId: string) => {
    setCards((current) => current.filter((card) => card.id !== cardId));
  }, []);

  const value = useMemo<ReadingActionsValue>(
    () => ({
      ageMode,
      actions,
      cards,
      speaking,
      activeWord,
      readAlongWords,
      busyKey,
      run,
      stopSpeaking,
      dismiss,
      readAlongActive,
      readAlongPreparing,
      revealReadAlong,
      registerReadAlongReveal,
      startReadAlong,
      speakSelection,
      toggleReadAlong,
      continueReadAlong,
      registerReadAlongSource,
    }),
    [
      actions,
      activeWord,
      ageMode,
      busyKey,
      cards,
      continueReadAlong,
      dismiss,
      readAlongActive,
      readAlongPreparing,
      readAlongWords,
      registerReadAlongReveal,
      revealReadAlong,
      registerReadAlongSource,
      run,
      speakSelection,
      speaking,
      startReadAlong,
      stopSpeaking,
      toggleReadAlong,
    ],
  );

  return (
    <ReadingActionsContext.Provider value={value}>
      {children}
    </ReadingActionsContext.Provider>
  );
}
