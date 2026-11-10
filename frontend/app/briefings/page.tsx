"use client";

import {
  BriefingPanel,
  useBriefing,
} from "./BriefingComponents";

export default function BriefingsPage() {
  const morningBriefing = useBriefing("morning");
  const eveningBriefing = useBriefing("evening");

  return (
    <>
      <a className="skip-link" href="#main-content">Skip to content</a>
      <main className="page-container static-section" id="main-content" tabIndex={-1}>
        <header className="section-heading static-section-heading">
          <div className="static-heading-copy">
            <p className="section-index">AI RADAR / DAILY INTELLIGENCE</p>
            <h1>Briefings</h1>
          </div>
        </header>
        <p className="quiet-state briefings-intro">
          Daily AI developments, with the morning baseline and evening comparison presented as generated.
        </p>
        <section aria-label="Morning and evening briefings">
          <h2 className="sr-only">Briefing snapshots</h2>
          <div className="briefing-grid">
            <BriefingPanel
              kind="morning"
              title="Morning briefing"
              description="The morning snapshot establishes the baseline of important AI developments for the day."
              requestState={morningBriefing.requestState}
              onRetry={morningBriefing.retry}
              headingLevel="h3"
            />
            <BriefingPanel
              kind="evening"
              title="Evening briefing"
              description="The evening snapshot reports on developments in relation to the morning baseline."
              requestState={eveningBriefing.requestState}
              onRetry={eveningBriefing.retry}
              headingLevel="h3"
            />
          </div>
        </section>
      </main>
      <footer className="site-footer">
        <div className="page-container footer-container">
          <span className="footer-copy">AI Radar — local-first AI intelligence.</span>
        </div>
      </footer>
    </>
  );
}
