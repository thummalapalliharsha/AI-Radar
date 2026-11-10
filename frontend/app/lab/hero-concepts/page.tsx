"use client";

import dynamic from "next/dynamic";

const HeroConceptsCanvas = dynamic(
  () => import("./HeroConceptsCanvas").then((module) => module.HeroConceptsCanvas),
  { ssr: false, loading: () => <div className="concept-loading">Loading exploratory concepts…</div> },
);

const concepts = [
  {
    key: "A",
    name: "Layered Intelligence Core",
    meaning: "Intelligence + synthesis + internal computation",
    geometry: "Nested translucent shells around a restrained computational center",
  },
  {
    key: "B",
    name: "Signal Architecture Core",
    meaning: "Signal + information flow + recency + synthesis",
    geometry: "Partial bands and engineered pathways converging on a central focal point",
  },
  {
    key: "C",
    name: "Modular Intelligence Core",
    meaning: "Models + agents + orchestration + intelligence system",
    geometry: "Coordinated physical modules organized around a central hierarchy",
  },
] as const;

export default function HeroConceptsLab() {
  return (
    <main className="concept-page">
      <header className="concept-header">
        <p className="concept-eyebrow">AI Radar / B.5 exploration gate</p>
        <h1>Intelligence Core concepts</h1>
        <p className="concept-intro">
          Three materially distinct exploratory interpretations for human comparison. These are isolated prototypes, not the production hero.
        </p>
      </header>

      <section className="concept-grid" aria-label="Exploratory hero concepts">
        {concepts.map((concept, index) => (
          <article className="concept-card" key={concept.key}>
            <div className="concept-visual" aria-label={`${concept.name} exploratory 3D prototype`}>
              <HeroConceptsCanvas concept={index as 0 | 1 | 2} />
            </div>
            <div className="concept-copy">
              <p className="concept-index">CONCEPT {concept.key}</p>
              <h2>{concept.name}</h2>
              <p>{concept.geometry}</p>
              <p className="concept-meaning">Meaning: {concept.meaning}</p>
            </div>
          </article>
        ))}
      </section>

      <footer className="concept-footer">
        <p>Exploration only. No concept is selected or frozen.</p>
        <p>Review silhouette, material realism, lighting, depth, typography relationship, mobile potential, fallback potential, and performance risk.</p>
      </footer>

      <style jsx>{`
        .concept-page { min-height: 100dvh; padding: 48px clamp(20px, 4vw, 64px) 72px; background: #05070b; color: #f4f7fb; font-family: var(--font-ui), sans-serif; }
        .concept-header { max-width: 760px; margin: 0 auto 48px; }
        .concept-eyebrow, .concept-index { margin: 0 0 14px; color: #4de7e1; font-size: 0.7rem; letter-spacing: 0.14em; text-transform: uppercase; }
        h1, h2 { font-family: var(--font-editorial), Georgia, serif; font-weight: 400; letter-spacing: -0.025em; }
        h1 { margin: 0; font-size: clamp(3rem, 7vw, 6.5rem); line-height: 0.95; }
        .concept-intro { max-width: 58ch; margin: 22px 0 0; color: #8b95a7; line-height: 1.6; }
        .concept-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: clamp(18px, 3vw, 40px); max-width: 1600px; margin: 0 auto; }
        .concept-card { min-width: 0; border-top: 1px solid rgba(244, 247, 251, 0.12); background: #0a0f17; }
        .concept-visual { height: clamp(280px, 32vw, 480px); border-bottom: 1px solid rgba(244, 247, 251, 0.08); }
        .concept-copy { padding: 22px clamp(16px, 2vw, 28px) 28px; }
        .concept-copy h2 { margin: 0 0 14px; font-size: clamp(1.8rem, 3vw, 2.8rem); line-height: 1; }
        .concept-copy p:not(.concept-index):not(.concept-meaning) { margin: 0; color: #c7d0dd; line-height: 1.55; }
        .concept-meaning { margin: 18px 0 0; color: #8b95a7; font-size: 0.86rem; line-height: 1.5; }
        .concept-footer { max-width: 1600px; margin: 40px auto 0; padding-top: 18px; border-top: 1px solid rgba(244, 247, 251, 0.08); color: #8b95a7; font-size: 0.82rem; line-height: 1.6; }
        .concept-footer p { margin: 0 0 6px; }
        .concept-loading { display: grid; place-items: center; height: 100%; color: #8b95a7; font-size: 0.85rem; }
        @media (max-width: 900px) { .concept-grid { grid-template-columns: 1fr; max-width: 720px; } .concept-visual { height: min(72vw, 480px); } }
        @media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; transition: none !important; } }
      `}</style>
    </main>
  );
}
