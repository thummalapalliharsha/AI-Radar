import { SearchPanel } from "./SearchPanel";

export default function SearchPage() {
  return (
    <>
      <a className="skip-link" href="#main-content">Skip to content</a>
      <main className="page-container static-section" id="main-content" tabIndex={-1}>
        <header className="section-heading static-section-heading">
          <div className="static-heading-copy">
            <p className="section-index">AI RADAR / GROUNDED RETRIEVAL</p>
            <h1>Search AI Radar</h1>
          </div>
        </header>
        <p className="quiet-state search-page-intro">
          Ask a question about AI. Answers and source articles come from the AI Radar collection.
        </p>
        <section className="search-page-panel" aria-label="Search verified AI Radar sources">
          <h2 className="sr-only">Ask AI Radar</h2>
          <SearchPanel headingLevel="h2" />
          <p className="quiet-state search-note">AI-related questions only.</p>
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
