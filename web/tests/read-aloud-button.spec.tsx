import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

/**
 * The reader's read-aloud control, plus the "where is it reading?" action.
 *
 * Read-along can be on a different page from the one on screen, and the cursor
 * deliberately does not drag the reader there — so the toolbar has to offer a
 * way to go to it.
 */

const state = vi.hoisted(() => ({
  readAlongActive: false,
  readAlongPreparing: false,
  speaking: false,
  busyKey: "",
  revealed: 0,
  toggled: 0,
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

vi.mock("@/components/reading/reading-actions-context", () => ({
  useReadingActions: () => ({
    ageMode: "default",
    actions: [{ key: "read_aloud:read", label: "Read aloud", needsSelection: false }],
    busyKey: state.busyKey,
    speaking: state.speaking,
    readAlongActive: state.readAlongActive,
    readAlongPreparing: state.readAlongPreparing,
    toggleReadAlong: () => {
      state.toggled += 1;
    },
    revealReadAlong: () => {
      state.revealed += 1;
    },
  }),
}));

const { ReadAloudButton } = await import(
  "@/components/reading/workspace/ReadAloudButton"
);

describe("read-aloud button", () => {
  beforeEach(() => {
    state.readAlongActive = false;
    state.readAlongPreparing = false;
    state.speaking = false;
    state.busyKey = "";
    state.revealed = 0;
    state.toggled = 0;
  });

  it("starts read-along when pressed", () => {
    render(<ReadAloudButton />);
    fireEvent.click(screen.getByLabelText("Read aloud from here"));
    expect(state.toggled).toBe(1);
  });

  it("has no locate action until read-along is running", () => {
    render(<ReadAloudButton />);
    expect(screen.queryByLabelText("Go to reading position")).toBeNull();
  });

  it("offers to go to the reading position once running", () => {
    state.readAlongActive = true;
    render(<ReadAloudButton />);

    fireEvent.click(screen.getByLabelText("Go to reading position"));

    expect(state.revealed).toBe(1);
  });

  it("stays pressable while the audio is being prepared, so the wait can be cancelled", () => {
    state.readAlongActive = true;
    state.readAlongPreparing = true;
    render(<ReadAloudButton />);

    const stop = screen.getByLabelText("Stop reading aloud");
    expect(stop).not.toBeDisabled();
    fireEvent.click(stop);
    expect(state.toggled).toBe(1);
  });
});
