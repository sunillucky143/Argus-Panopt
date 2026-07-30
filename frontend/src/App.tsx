import { useEffect, useState } from "react";

type ApiState = "checking" | "ready" | "unavailable";

interface HealthResponse {
  status: "ready";
  service: "api";
  version: string;
}

const isHealthResponse = (value: unknown): value is HealthResponse => {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  return (
    "status" in value &&
    value.status === "ready" &&
    "service" in value &&
    value.service === "api" &&
    "version" in value &&
    typeof value.version === "string"
  );
};

export function App() {
  const [apiState, setApiState] = useState<ApiState>("checking");

  useEffect(() => {
    const controller = new AbortController();

    const checkHealth = async () => {
      try {
        const response = await fetch("/api/health/ready", {
          headers: { Accept: "application/json" },
          signal: controller.signal,
        });
        const payload: unknown = await response.json();
        setApiState(
          response.ok && isHealthResponse(payload) ? "ready" : "unavailable",
        );
      } catch (error: unknown) {
        if (!(error instanceof DOMException && error.name === "AbortError")) {
          setApiState("unavailable");
        }
      }
    };

    void checkHealth();
    return () => {
      controller.abort();
    };
  }, []);

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Projects">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            A
          </div>
          <div>
            <strong>Argus Panopt</strong>
            <span>Private intelligence</span>
          </div>
        </div>

        <button className="new-project" type="button" disabled>
          <span aria-hidden="true">＋</span>
          New project
        </button>

        <nav className="project-list" aria-label="Project list">
          <p className="section-label">Your projects</p>
          <div className="empty-projects">
            <span aria-hidden="true">◇</span>
            <p>
              Projects will appear here after the secure workspace is enabled.
            </p>
          </div>
        </nav>

        <div className="privacy-note">
          <span className="lock" aria-hidden="true">
            ●
          </span>
          <div>
            <strong>Self-hosted</strong>
            <span>Your data stays in this deployment.</span>
          </div>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <span
              className={`status-dot status-${apiState}`}
              aria-hidden="true"
            />
            <span className="status-text" aria-live="polite">
              {apiState === "checking" && "Checking local services"}
              {apiState === "ready" && "Local services ready"}
              {apiState === "unavailable" && "API unavailable"}
            </span>
          </div>
          <span className="phase-badge">Phase 0</span>
        </header>

        <section className="hero" aria-labelledby="welcome-title">
          <div className="hero-symbol" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <p className="eyebrow">Private by design</p>
          <h1 id="welcome-title">
            Understand documents without sending them away.
          </h1>
          <p className="lede">
            Argus Panopt will turn your PDFs, Word documents, and spreadsheets
            into a secure, cited knowledge workspace—using models that run
            entirely here.
          </p>

          <div className="feature-grid" aria-label="Platform principles">
            <article>
              <span aria-hidden="true">01</span>
              <h2>Grounded answers</h2>
              <p>
                Every answer will point back to the exact page, table, or cell.
              </p>
            </article>
            <article>
              <span aria-hidden="true">02</span>
              <h2>Local inference</h2>
              <p>
                Generation, embeddings, reranking, and OCR stay inside the
                stack.
              </p>
            </article>
            <article>
              <span aria-hidden="true">03</span>
              <h2>Controlled retention</h2>
              <p>Choose persistent or ephemeral handling for each project.</p>
            </article>
          </div>

          <p className="coming-soon">
            Document workspaces arrive in Phase 2. This shell verifies the
            secure deployment foundation.
          </p>
        </section>
      </main>
    </div>
  );
}
