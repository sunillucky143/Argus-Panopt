import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("renders the privacy-first shell and reports a healthy API", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            status: "ready",
            service: "api",
            version: "0.1.0",
          }),
      }),
    );

    render(<App />);

    expect(
      screen.getByRole("heading", {
        name: "Understand documents without sending them away.",
      }),
    ).toBeInTheDocument();
    expect(await screen.findByText("Local services ready")).toBeInTheDocument();
  });

  it("reports an unavailable API without exposing error detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new Error("sensitive network detail")),
    );

    render(<App />);

    expect(await screen.findByText("API unavailable")).toBeInTheDocument();
    expect(
      screen.queryByText("sensitive network detail"),
    ).not.toBeInTheDocument();
  });

  it("rejects malformed health responses", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(null),
      }),
    );

    render(<App />);

    expect(await screen.findByText("API unavailable")).toBeInTheDocument();
  });

  it("cancels its health request when unmounted", async () => {
    const fetchMock = vi.fn(
      (_input: RequestInfo | URL, init?: RequestInit): Promise<Response> =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("request cancelled", "AbortError"));
          });
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const { unmount } = render(<App />);
    unmount();

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledOnce();
    });
  });
});
