"use client";

import { motion, type MotionProps, type Variants } from "framer-motion";
import dynamic from "next/dynamic";
import { useEffect, useState, useSyncExternalStore } from "react";
import {
  BriefingPanel,
  SinceMorningStatus,
  useBriefing,
} from "./briefings/BriefingComponents";
import { SearchPanel } from "./search/SearchPanel";

const PlanetaryHero = dynamic(
  () => import("./PlanetaryHero").then((module) => module.PlanetaryHero),
  { ssr: false, loading: () => <div className="planetary-fallback" aria-hidden="true" /> },
);

const revealVariants: Variants = {
  hidden: { opacity: 0, y: 16 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: "easeOut", staggerChildren: 0.08 },
  },
};

const revealChildVariants: Variants = {
  hidden: { opacity: 0, y: 16 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" } },
};

const heroRevealVariants: Variants = {
  hidden: { opacity: 0, y: 16 },
  visible: (order: number) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: "easeOut", delay: order * 0.08 },
  }),
};

const heroContainerVariants: Variants = {
  hidden: {},
  visible: {},
};

type MarkedMotionProps = MotionProps & { "data-motion-target": "" };

const immediateMotionProps: MarkedMotionProps = {
  initial: false,
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0 },
  "data-motion-target": "",
};

function subscribeToReducedMotion(onChange: () => void) {
  const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  mediaQuery.addEventListener("change", onChange);
  return () => mediaQuery.removeEventListener("change", onChange);
}

function getReducedMotionPreference() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function useHydratedReducedMotion() {
  return useSyncExternalStore(
    subscribeToReducedMotion,
    getReducedMotionPreference,
    () => false,
  );
}

function revealMotionProps(reducedMotion: boolean): MarkedMotionProps {
  return reducedMotion
    ? immediateMotionProps
    : {
        "data-motion-target": "",
        initial: { opacity: 0, y: 16 },
        whileInView: { opacity: 1, y: 0 },
        viewport: { once: true, margin: "-10% 0px" },
        transition: { duration: 0.5, ease: "easeOut" },
      };
}

function staggerMotionProps(reducedMotion: boolean): MarkedMotionProps {
  return reducedMotion
    ? immediateMotionProps
    : {
        "data-motion-target": "",
        initial: "hidden",
        whileInView: "visible",
        viewport: { once: true, margin: "-10% 0px" },
        variants: revealVariants,
      };
}

function staggerChildProps(reducedMotion: boolean): MarkedMotionProps {
  return reducedMotion
    ? immediateMotionProps
    : { "data-motion-target": "", variants: revealChildVariants };
}

function heroMotionProps(reducedMotion: boolean): MarkedMotionProps {
  return reducedMotion
    ? immediateMotionProps
    : {
        "data-motion-target": "",
        initial: "hidden",
        whileInView: "visible",
        viewport: { once: true, margin: "-10% 0px" },
        variants: heroContainerVariants,
      };
}

function heroChildProps(reducedMotion: boolean, order: number): MarkedMotionProps {
  return reducedMotion
    ? immediateMotionProps
    : { "data-motion-target": "", variants: heroRevealVariants, custom: order };
}

const timeRanges = [
  { title: "Last 24 hours", id: "last-24-hours", label: "24H", tone: "day", window: "24h" as const },
  { title: "This week", id: "this-week", label: "7D", tone: "week", window: "7d" as const },
];

type ApiStory = {
  headline: string;
  summary: string;
  category: string;
  source: string;
  published_at: string;
  url: string;
  why_it_matters: string | null;
};

type StoryRequestState =
  | { status: "loading" }
  | { status: "error" }
  | { status: "loaded"; stories: ApiStory[] };

const apiBaseUrl = (
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"
).replace(/\/+$/, "");
const storyRefreshIntervalMs = 5 * 60 * 1000;

function isApiStory(value: unknown): value is ApiStory {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  const story = value as Record<string, unknown>;
  if (
    typeof story.headline !== "string" ||
    typeof story.summary !== "string" ||
    typeof story.category !== "string" ||
    typeof story.source !== "string" ||
    typeof story.published_at !== "string" ||
    typeof story.url !== "string" ||
    (typeof story.why_it_matters !== "string" && story.why_it_matters !== null) ||
    !Number.isFinite(Date.parse(story.published_at))
  ) {
    return false;
  }

  try {
    const url = new URL(story.url);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

function isApiStoryList(value: unknown): value is ApiStory[] {
  return Array.isArray(value) && value.every(isApiStory);
}

function StoryRows({ window, reducedMotion }: { window: "24h" | "7d"; reducedMotion: boolean }) {
  const [requestState, setRequestState] = useState<StoryRequestState>({
    status: "loading",
  });

  useEffect(() => {
    const controller = new AbortController();
    let requestInFlight = false;
    let hasSettled = false;

    async function loadStories(isInitialRequest: boolean) {
      if (requestInFlight) {
        return;
      }
      requestInFlight = true;

      if (isInitialRequest) {
        setRequestState({ status: "loading" });
      }

      try {
        const response = await fetch(
          `${apiBaseUrl}/api/stories?window=${window}&limit=20`,
          { signal: controller.signal },
        );
        if (!response.ok) {
          throw new Error(`Stories request failed with HTTP ${response.status}.`);
        }

        const payload: unknown = await response.json();
        if (!isApiStoryList(payload)) {
          throw new Error("Stories response did not match the expected format.");
        }
        setRequestState({ status: "loaded", stories: payload });
        hasSettled = true;
      } catch {
        if (controller.signal.aborted) {
          return;
        }
        if (!hasSettled) {
          setRequestState({ status: "error" });
          hasSettled = true;
        }
      } finally {
        requestInFlight = false;
      }
    }

    void loadStories(true);
    const refreshInterval = globalThis.setInterval(() => {
      void loadStories(false);
    }, storyRefreshIntervalMs);

    return () => {
      controller.abort();
      globalThis.clearInterval(refreshInterval);
    };
  }, [window]);

  return (
    <div
      className="awaiting-table"
      role="group"
      aria-label={`AI developments from the last ${window === "24h" ? "24 hours" : "7 days"}`}
    >
      {requestState.status === "loading" ? (
        <div className="signals-loading" role="status" aria-label="Loading verified AI stories">
          <span className="signals-loading-line signals-loading-line-wide" />
          <span className="signals-loading-line" />
          <span className="signals-loading-line signals-loading-line-short" />
        </div>
      ) : requestState.status === "error" ? (
        <p className="story-status quiet-state signal-state" role="alert">
          Unable to load stories for the last {window === "24h" ? "24 hours" : "7 days"}.
        </p>
      ) : requestState.stories.length === 0 ? (
        <p className="story-status quiet-state signal-state" role="status">
          No verified AI stories are available in the last {window === "24h" ? "24 hours" : "7 days"}.
        </p>
      ) : (
        <motion.ul
          className="signals-list"
          initial={reducedMotion ? false : "hidden"}
          whileInView={reducedMotion ? undefined : "visible"}
          viewport={{ once: true, margin: "-8% 0px" }}
          variants={{ hidden: {}, visible: { transition: { staggerChildren: 0.07 } } }}
        >
          {requestState.stories.map((story, index) => (
            <motion.li
              className={`signal-card${index === 0 ? " signal-card-lead" : ""}`}
              key={story.url}
              variants={reducedMotion ? undefined : { hidden: { opacity: 0, y: 14 }, visible: { opacity: 1, y: 0, transition: { duration: 0.45, ease: "easeOut" } } }}
            >
              <div className="signal-card-header">
                <span className="signal-index" aria-hidden="true">{String(index + 1).padStart(2, "0")}</span>
                <span className="signal-category">{story.category}</span>
              </div>
              <a
                className="story-headline"
                href={story.url}
                target="_blank"
                rel="noopener noreferrer"
                aria-label={`Read ${story.headline} from ${story.source} (opens in a new tab)`}
              >
                {story.headline}
              </a>
              <p className="signal-summary">{story.summary}</p>
              <div className="signal-card-meta">
                <span className="story-source">
                  <span className="sr-only">Source: </span>
                  {story.source}
                </span>
                <time className="story-published" dateTime={story.published_at}>
                  <span className="sr-only">Published: </span>
                  {new Date(story.published_at).toLocaleString(undefined, {
                    dateStyle: "medium",
                    timeStyle: "short",
                  })}
                </time>
              </div>
              {story.why_it_matters && (
                <p className="signal-why">
                  <span className="signal-why-label">WHY IT MATTERS</span>
                  {story.why_it_matters}
                </p>
              )}
            </motion.li>
          ))}
        </motion.ul>
      )}
    </div>
  );
}

function CategoryStoryRows({ category }: { category: "AI Research" | "AI Models" | "AI Agents" }) {
  const [requestState, setRequestState] = useState<StoryRequestState>({
    status: "loading",
  });

  useEffect(() => {
    const controller = new AbortController();
    let requestInFlight = false;
    let hasSettled = false;
    const searchParams = new URLSearchParams({
      window: "7d",
      limit: "5",
      category,
    });

    async function loadStories(isInitialRequest: boolean) {
      if (requestInFlight) {
        return;
      }
      requestInFlight = true;

      if (isInitialRequest) {
        setRequestState({ status: "loading" });
      }

      try {
        const response = await fetch(
          `${apiBaseUrl}/api/stories?${searchParams.toString()}`,
          { signal: controller.signal },
        );
        if (!response.ok) {
          throw new Error(`Stories request failed with HTTP ${response.status}.`);
        }

        const payload: unknown = await response.json();
        if (!isApiStoryList(payload)) {
          throw new Error("Stories response did not match the expected format.");
        }
        setRequestState({ status: "loaded", stories: payload });
        hasSettled = true;
      } catch {
        if (!controller.signal.aborted && !hasSettled) {
          setRequestState({ status: "error" });
          hasSettled = true;
        }
      } finally {
        requestInFlight = false;
      }
    }

    void loadStories(true);
    const refreshInterval = globalThis.setInterval(() => {
      void loadStories(false);
    }, storyRefreshIntervalMs);

    return () => {
      controller.abort();
      globalThis.clearInterval(refreshInterval);
    };
  }, [category]);

  if (requestState.status === "loading") {
    return <p className="quiet-state editorial-stories-state" role="status">Loading verified {category} stories…</p>;
  }
  if (requestState.status === "error") {
    return <p className="quiet-state editorial-stories-state" role="alert">Unable to load verified {category} stories.</p>;
  }
  if (requestState.stories.length === 0) {
    return <p className="quiet-state editorial-stories-state" role="status">No verified {category} stories are available in the last seven days.</p>;
  }

  return (
    <ol className="briefing-stories editorial-stories" aria-label={`Verified ${category} stories from the last seven days`}>
      {requestState.stories.map((story, index) => (
        <li className="briefing-story" key={story.url}>
          <span className="briefing-story-index" aria-hidden="true">
            {String(index + 1).padStart(2, "0")}
          </span>
          <div className="briefing-story-content">
            <h4>
              <a href={story.url} target="_blank" rel="noopener noreferrer">
                {story.headline}
              </a>
            </h4>
            <p className="briefing-story-meta">
              <span>{story.source}</span>
              <span>{story.category}</span>
              <time dateTime={story.published_at}>
                {new Date(story.published_at).toLocaleString(undefined, {
                  dateStyle: "medium",
                  timeStyle: "short",
                })}
              </time>
            </p>
            <p className="quiet-state briefing-story-summary">{story.summary}</p>
            {story.why_it_matters && (
              <p className="briefing-story-why">
                <span className="briefing-story-label">WHY IT MATTERS</span>
                {story.why_it_matters}
              </p>
            )}
          </div>
        </li>
      ))}
    </ol>
  );
}

export default function Home() {
  const [footerInView, setFooterInView] = useState(false);
  const reducedMotion = useHydratedReducedMotion();
  const morningBriefing = useBriefing("morning");
  const eveningBriefing = useBriefing("evening");

  useEffect(() => {
    const footer = document.querySelector(".site-footer");
    if (!footer) return;

    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) setFooterInView(true);
    });
    observer.observe(footer);
    return () => observer.disconnect();
  }, []);

  return (
    <>
      <a className="skip-link" href="#main-content">Skip to content</a>
      <main id="main-content" tabIndex={-1}>
        <section className="page-container hero-composition hero-foundation" aria-labelledby="hero-title">
          <motion.div className="intro-copy" {...heroMotionProps(reducedMotion)}>
            <motion.p className="eyebrow" {...heroChildProps(reducedMotion, 0)}>AI RADAR · GLOBAL INTELLIGENCE</motion.p>
            <hr className="hero-rule" aria-hidden="true" />
            <motion.h1 id="hero-title" {...heroChildProps(reducedMotion, 1)}>Intelligence in orbit</motion.h1>
            <motion.p className="intro-title" id="intro-title" {...heroChildProps(reducedMotion, 2)}>Welcome to the AI world</motion.p>
            <motion.p className="intro-support" {...heroChildProps(reducedMotion, 3)}>Explore the discoveries, models, and agents reshaping the AI landscape.</motion.p>
            <motion.div className="hero-actions" {...heroChildProps(reducedMotion, 4)}>
              <a className="hero-action hero-action-primary" href="#latest">Explore Latest</a>
              <a className="hero-action hero-action-secondary" href="#search">Ask AI Radar <span aria-hidden="true">↗</span></a>
            </motion.div>
          </motion.div>
          <div className="hero-planet">
            <PlanetaryHero />
          </div>
        </section>

        <motion.section className="page-container latest" id="latest" aria-labelledby="latest-title" {...revealMotionProps(reducedMotion)}>
          <motion.div className="section-heading" {...staggerMotionProps(reducedMotion)}>
            <motion.h2 id="latest-title" {...staggerChildProps(reducedMotion)}>Latest AI</motion.h2>
            <motion.span className="section-index" {...staggerChildProps(reducedMotion)}>01 / Recency</motion.span>
          </motion.div>

          <section className="time-chapter" aria-labelledby="since-morning-title">
            <p className="time-label time-label-now">NOW</p>
            <h3 id="since-morning-title">Since this morning</h3>
            <SinceMorningStatus
              morningState={morningBriefing.requestState}
              eveningState={eveningBriefing.requestState}
            />
          </section>

          {timeRanges.map((range) => (
            <section className="time-chapter" id={range.id} aria-labelledby={`${range.id}-title`} key={range.id}>
              <p className={`time-label time-label-${range.tone}`}>{range.label}</p>
              <h3 id={`${range.id}-title`}>{range.title}</h3>
              <StoryRows window={range.window} reducedMotion={reducedMotion} />
            </section>
          ))}
          <p className="latest-footnote">Stories shown here come from verified AI Radar sources.</p>
        </motion.section>

        <section className="page-container editorial-chapters" aria-label="Research, models, and agents">
          <motion.article className="editorial-chapter" id="research" {...staggerMotionProps(reducedMotion)}>
            <motion.p className="section-index" {...staggerChildProps(reducedMotion)}>01 / RESEARCH</motion.p>
            <div className="editorial-chapter-copy">
              <motion.h2 {...staggerChildProps(reducedMotion)}>Research</motion.h2>
              <div className="editorial-anchor-context">
                <motion.p className="quiet-state" {...staggerChildProps(reducedMotion)}>Track important research, papers, methods, and scientific developments shaping AI.</motion.p>
                <motion.hr className="editorial-anchor-rule editorial-anchor-rule-research" aria-hidden="true" {...staggerChildProps(reducedMotion)} />
                <motion.p className="editorial-keywords" {...staggerChildProps(reducedMotion)}>PAPERS · METHODS · BENCHMARKS</motion.p>
              </div>
            </div>
            <CategoryStoryRows category="AI Research" />
          </motion.article>
          <motion.article className="editorial-chapter editorial-chapter-aligned" id="models" {...staggerMotionProps(reducedMotion)}>
            <motion.p className="section-index" {...staggerChildProps(reducedMotion)}>02 / MODELS</motion.p>
            <div className="editorial-chapter-copy">
              <motion.h2 {...staggerChildProps(reducedMotion)}>Models</motion.h2>
              <div className="editorial-anchor-context">
                <motion.p className="quiet-state" {...staggerChildProps(reducedMotion)}>Follow major model developments, capabilities, releases, and changes across the AI landscape.</motion.p>
                <motion.hr className="editorial-anchor-rule editorial-anchor-rule-models" aria-hidden="true" {...staggerChildProps(reducedMotion)} />
                <motion.p className="editorial-keywords" {...staggerChildProps(reducedMotion)}>RELEASES · CAPABILITIES · COMPARISONS</motion.p>
              </div>
            </div>
            <CategoryStoryRows category="AI Models" />
          </motion.article>
          <motion.article className="editorial-chapter editorial-chapter-narrow" id="agents" {...staggerMotionProps(reducedMotion)}>
            <motion.p className="section-index" {...staggerChildProps(reducedMotion)}>03 / AGENTS</motion.p>
            <div className="editorial-chapter-copy">
              <motion.h2 {...staggerChildProps(reducedMotion)}>Agents</motion.h2>
              <div className="editorial-anchor-context">
                <motion.p className="quiet-state" {...staggerChildProps(reducedMotion)}>Track developments in AI agents, agent systems, workflows, and autonomous capabilities.</motion.p>
                <motion.hr className="editorial-anchor-rule editorial-anchor-rule-agents" aria-hidden="true" {...staggerChildProps(reducedMotion)} />
                <motion.p className="editorial-keywords" {...staggerChildProps(reducedMotion)}>WORKFLOWS · TOOLING · AUTONOMY</motion.p>
              </div>
            </div>
            <CategoryStoryRows category="AI Agents" />
          </motion.article>
        </section>

        <motion.section className="page-container static-section" id="search" aria-labelledby="search-title" {...staggerMotionProps(reducedMotion)}>
          <div className="section-heading static-section-heading">
            <div className="static-heading-copy">
              <motion.p className="section-index" {...staggerChildProps(reducedMotion)}>04 / SEARCH</motion.p>
              <motion.h2 id="search-title" {...staggerChildProps(reducedMotion)}>Search AI Radar</motion.h2>
            </div>
          </div>
          <SearchPanel headingLevel="h3" />
          <motion.p className="quiet-state search-note" {...staggerChildProps(reducedMotion)}>Search verified AI Radar sources.</motion.p>
          <motion.p className="quiet-state search-note" {...staggerChildProps(reducedMotion)}>AI-related questions only.</motion.p>
        </motion.section>

        <motion.section className="page-container static-section" id="briefings" aria-labelledby="briefings-title" {...revealMotionProps(reducedMotion)}>
          <motion.div className="section-heading static-section-heading" {...staggerMotionProps(reducedMotion)}>
            <div className="static-heading-copy">
              <motion.p className="section-index" {...staggerChildProps(reducedMotion)}>05 / BRIEFINGS</motion.p>
              <motion.h2 id="briefings-title" {...staggerChildProps(reducedMotion)}>Briefings</motion.h2>
            </div>
          </motion.div>
          <div className="briefing-grid">
            <BriefingPanel
              kind="morning"
              title="Morning briefing"
              description="Establishes the baseline of important AI developments for the day."
              requestState={morningBriefing.requestState}
              onRetry={morningBriefing.retry}
            />
            <BriefingPanel
              kind="evening"
              title="Evening briefing"
              description="Compares new developments against the morning baseline."
              requestState={eveningBriefing.requestState}
              onRetry={eveningBriefing.retry}
            />
          </div>
        </motion.section>
      </main>
      <motion.footer
        className="site-footer"
        {...revealMotionProps(reducedMotion)}
        animate={footerInView || reducedMotion ? { opacity: 1, y: 0 } : undefined}
      >
        <div className="page-container footer-container">
          <span className="footer-copy">AI Radar — local-first AI intelligence.</span>
        </div>
      </motion.footer>
    </>
  );
}
