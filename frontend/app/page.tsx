"use client";

import { motion, type MotionProps, type Variants } from "framer-motion";
import dynamic from "next/dynamic";
import { useEffect, useRef, useState, useSyncExternalStore, type FormEvent } from "react";

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

function StoryRows({ window }: { window: "24h" | "7d" }) {
  const [requestState, setRequestState] = useState<StoryRequestState>({
    status: "loading",
  });

  useEffect(() => {
    const controller = new AbortController();

    async function loadStories() {
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
      } catch {
        if (controller.signal.aborted) {
          return;
        }
        setRequestState({ status: "error" });
      }
    }

    void loadStories();
    return () => controller.abort();
  }, [window]);

  return (
    <div
      className="awaiting-table"
      role="group"
      aria-label={`AI developments from the last ${window === "24h" ? "24 hours" : "7 days"}`}
    >
      <div className="awaiting-header" aria-hidden="true">
        <span>Headline</span>
        <span>Source</span>
        <span>Published</span>
        <span>Category</span>
      </div>
      {requestState.status === "loading" ? (
        <p className="story-status quiet-state" role="status">Loading verified AI stories…</p>
      ) : requestState.status === "error" ? (
        <p className="story-status quiet-state" role="alert">
          Unable to load stories for the last {window === "24h" ? "24 hours" : "7 days"}.
        </p>
      ) : requestState.stories.length === 0 ? (
        <p className="story-status quiet-state" role="status">
          No verified AI stories are available in the last {window === "24h" ? "24 hours" : "7 days"}.
        </p>
      ) : (
        <ul className="awaiting-list">
          {requestState.stories.map((story) => (
            <li className="awaiting-row story-row" key={story.url}>
              <a
                className="story-headline"
                href={story.url}
                target="_blank"
                rel="noopener noreferrer"
                aria-label={`Read ${story.headline} from ${story.source} (opens in a new tab)`}
              >
                {story.headline}
              </a>
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
              <span className="awaiting-tag story-category">
                <span className="sr-only">Category: </span>
                {story.category}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

type BriefingStory = {
  title: string;
  summary: string;
  source?: string | null;
  published?: string | null;
  category?: string | null;
  url?: string | null;
  why_it_matters?: string | null;
};

type BriefingData = {
  available: true;
  baseline_missing: boolean;
  generated_at: string;
  summary: string;
  stories: BriefingStory[];
  current_local_date: string;
  generated_local_date: string;
  baseline_local_date: string | null;
  is_current_day: boolean;
  is_current_comparison: boolean;
};

type BriefingRequestState =
  | { status: "loading" }
  | { status: "error" }
  | { status: "unavailable" }
  | { status: "loaded"; briefing: BriefingData };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isBriefingStory(value: unknown): value is BriefingStory {
  if (
    !isRecord(value) ||
    typeof value.title !== "string" ||
    !value.title.trim() ||
    typeof value.summary !== "string" ||
    !value.summary.trim()
  ) {
    return false;
  }

  for (const field of ["source", "category", "published", "why_it_matters"] as const) {
    if (value[field] !== undefined && value[field] !== null && typeof value[field] !== "string") {
      return false;
    }
  }

  if (typeof value.published === "string" && !Number.isFinite(Date.parse(value.published))) {
    return false;
  }

  if (value.url !== undefined && value.url !== null) {
    if (typeof value.url !== "string") {
      return false;
    }
    try {
      const url = new URL(value.url);
      if (url.protocol !== "http:" && url.protocol !== "https:") {
        return false;
      }
    } catch {
      return false;
    }
  }

  return true;
}

function isBriefingData(value: unknown): value is BriefingData | { available: false } {
  if (!isRecord(value)) {
    return false;
  }
  if (value.available === false) {
    return true;
  }
  return (
    value.available === true &&
    typeof value.baseline_missing === "boolean" &&
    typeof value.generated_at === "string" &&
    value.generated_at.length > 0 &&
    Number.isFinite(Date.parse(value.generated_at)) &&
    typeof value.summary === "string" &&
    value.summary.trim().length > 0 &&
    Array.isArray(value.stories) &&
    value.stories.every(isBriefingStory) &&
    typeof value.current_local_date === "string" &&
    typeof value.generated_local_date === "string" &&
    (typeof value.baseline_local_date === "string" || value.baseline_local_date === null) &&
    typeof value.is_current_day === "boolean" &&
    typeof value.is_current_comparison === "boolean"
  );
}

function useBriefing(kind: "morning" | "evening"): BriefingRequestState {
  const [requestState, setRequestState] = useState<BriefingRequestState>({
    status: "loading",
  });

  useEffect(() => {
    const controller = new AbortController();

    async function loadBriefing() {
      try {
        const response = await fetch(`${apiBaseUrl}/api/briefing/${kind}`, {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Briefing request failed with HTTP ${response.status}.`);
        }

        const payload: unknown = await response.json();
        if (!isBriefingData(payload)) {
          throw new Error("Briefing response did not match the expected format.");
        }
        if (payload.available === false) {
          setRequestState({ status: "unavailable" });
        } else {
          setRequestState({ status: "loaded", briefing: payload });
        }
      } catch {
        if (controller.signal.aborted) {
          return;
        }
        setRequestState({ status: "error" });
      }
    }

    void loadBriefing();
    return () => controller.abort();
  }, [kind]);

  return requestState;
}

function formatTimestamp(value: string): string {
  return new Date(value).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

function formatBriefingTimestamp(value: string): string {
  return `${new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "Asia/Kolkata",
  }).format(new Date(value))} IST`;
}

function SinceMorningStatus({
  morningState,
  eveningState,
}: {
  morningState: BriefingRequestState;
  eveningState: BriefingRequestState;
}) {
  const isLoading = morningState.status === "loading" || eveningState.status === "loading";
  if (isLoading) {
    return <p className="quiet-state">Loading today’s briefing status…</p>;
  }

  if (
    eveningState.status === "loaded" &&
    eveningState.briefing.is_current_comparison
  ) {
    return (
      <div className="since-morning-status">
        <p className="quiet-state">{eveningState.briefing.summary}</p>
        <p className="briefing-generated">
          Evening briefing generated{" "}
          <time dateTime={eveningState.briefing.generated_at}>
            {formatTimestamp(eveningState.briefing.generated_at)}
          </time>
        </p>
      </div>
    );
  }

  const hasError = morningState.status === "error" || eveningState.status === "error";
  if (hasError) {
    return <p className="quiet-state" role="alert">Unable to confirm today’s briefing status.</p>;
  }

  if (
    morningState.status === "loaded" &&
    morningState.briefing.is_current_day
  ) {
    const eveningMessage =
      eveningState.status === "loaded" && eveningState.briefing.is_current_day
        ? "No evening comparison for today’s morning baseline is available."
        : "No evening comparison has been generated for today.";
    return (
      <div className="since-morning-status">
        <p className="quiet-state">
          Today’s morning baseline is ready. {eveningMessage}
        </p>
        <p className="briefing-generated">
          Morning baseline generated{" "}
          <time dateTime={morningState.briefing.generated_at}>
            {formatTimestamp(morningState.briefing.generated_at)}
          </time>
        </p>
      </div>
    );
  }

  const message =
    morningState.status === "loaded" || eveningState.status === "loaded"
      ? "No current-day morning baseline is available yet."
      : "No briefing has been generated for today yet.";

  return <p className="quiet-state">{message}</p>;
}

function BriefingPanel({
  kind,
  title,
  description,
  requestState,
}: {
  kind: "morning" | "evening";
  title: string;
  description: string;
  requestState: BriefingRequestState;
}) {
  const stateMessage = {
    loading: `Loading ${kind} briefing…`,
    error: `Unable to load the ${kind} briefing.`,
    unavailable: `The ${kind} briefing is currently unavailable.`,
  } as const;

  const currentStateLabel = requestState.status === "loaded"
    ? kind === "morning"
      ? requestState.briefing.is_current_day
        ? "Current morning baseline"
        : "Historical morning snapshot — not today’s update"
      : requestState.briefing.is_current_comparison
        ? "Since this morning"
        : requestState.briefing.baseline_missing
          ? "Evening comparison unavailable"
          : requestState.briefing.is_current_day
            ? "No current evening comparison"
            : "Historical evening snapshot — not today’s update"
    : null;

  return (
    <article className={`briefing-block briefing-block-${kind}`}>
      <h3>{title}</h3>
      <p className="quiet-state">{description}</p>
      <div className="briefing-divider" aria-hidden="true" />
      {requestState.status === "loaded" ? (
        <>
          <p className="briefing-state" role="status">{currentStateLabel}</p>
          <p className="briefing-summary quiet-state">
            {requestState.briefing.summary}
          </p>
          <p className="briefing-generated">
            Generated{" "}
            <time dateTime={requestState.briefing.generated_at}>
              {formatBriefingTimestamp(requestState.briefing.generated_at)}
            </time>
          </p>
          {requestState.briefing.stories.length > 0 &&
          !(kind === "evening" && requestState.briefing.baseline_missing) ? (
            <ol className="briefing-stories" aria-label={`${kind} briefing stories`}>
              {requestState.briefing.stories.map((story, index) => (
                <li className="briefing-story" key={story.url || `${story.title}-${index}`}>
                  <span className="briefing-story-index" aria-hidden="true">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  <div className="briefing-story-content">
                    <h4>
                      {story.url ? (
                        <a
                          href={story.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          aria-label={`Read ${story.title} from ${story.source || "the original source"}`}
                        >
                          {story.title}
                        </a>
                      ) : (
                        story.title
                      )}
                    </h4>
                    <p className="briefing-story-meta">
                      {story.source && <span>{story.source}</span>}
                      {story.published && (
                        <time dateTime={story.published}>{formatTimestamp(story.published)}</time>
                      )}
                      {story.category && <span>{story.category}</span>}
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
          ) : (
            <p className="quiet-state briefing-empty-state" role="status">
              No stories are included in the {kind} briefing.
            </p>
          )}
        </>
      ) : (
        <p
          className={kind === "morning" ? "awaiting-tag" : "quiet-state briefing-empty-state"}
          role={requestState.status === "error" ? "alert" : "status"}
        >
          {stateMessage[requestState.status]}
        </p>
      )}
    </article>
  );
}

type SearchSource = {
  title: string;
  source: string;
  category: string;
  published: string;
  url: string | null;
  summary: string;
  why_it_matters: string;
};

type SearchResponse =
  | { status: "answered"; answer: string; sources: SearchSource[] }
  | { status: "no_relevant_information"; message: string }
  | { status: "unrelated"; message: string }
  | { status: "sources_only"; message: string; sources: SearchSource[] };

type SearchRequestState =
  | { status: "idle" }
  | { status: "typing" }
  | { status: "empty_query" }
  | { status: "loading" }
  | { status: "error" }
  | { status: "result"; query: string; response: SearchResponse };

function isSearchSource(value: unknown): value is SearchSource {
  if (!isRecord(value)) {
    return false;
  }
  if (
    typeof value.title !== "string" ||
    typeof value.source !== "string" ||
    typeof value.category !== "string" ||
    typeof value.published !== "string" ||
    typeof value.summary !== "string" ||
    typeof value.why_it_matters !== "string" ||
    (typeof value.url !== "string" && value.url !== null)
  ) {
    return false;
  }
  if (value.url === null) {
    return true;
  }
  try {
    const url = new URL(value.url);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

function isSearchResponse(value: unknown): value is SearchResponse {
  if (!isRecord(value) || typeof value.status !== "string") {
    return false;
  }

  if (value.status === "answered") {
    return (
      typeof value.answer === "string" &&
      value.answer.trim().length > 0 &&
      Array.isArray(value.sources) &&
      value.sources.length > 0 &&
      value.sources.every(isSearchSource)
    );
  }
  if (value.status === "no_relevant_information" || value.status === "unrelated") {
    return typeof value.message === "string" && value.message.trim().length > 0;
  }
  return (
    value.status === "sources_only" &&
    typeof value.message === "string" &&
    value.message.trim().length > 0 &&
    Array.isArray(value.sources) &&
    value.sources.length > 0 &&
    value.sources.every(isSearchSource)
  );
}

function SearchPanel() {
  const [query, setQuery] = useState("");
  const [requestState, setRequestState] = useState<SearchRequestState>({
    status: "idle",
  });
  const requestId = useRef(0);
  const activeController = useRef<AbortController | null>(null);

  useEffect(
    () => () => {
      requestId.current += 1;
      activeController.current?.abort();
    },
    [],
  );

  async function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const cleanQuery = query.trim();
    activeController.current?.abort();
    activeController.current = null;

    if (!cleanQuery) {
      requestId.current += 1;
      setRequestState({ status: "empty_query" });
      return;
    }

    const currentRequestId = ++requestId.current;
    const controller = new AbortController();
    activeController.current = controller;
    setRequestState({ status: "loading" });

    try {
      const searchUrl = new URL(`${apiBaseUrl}/api/search`);
      searchUrl.searchParams.set("q", cleanQuery);
      const response = await fetch(searchUrl, { signal: controller.signal });
      if (!response.ok) {
        throw new Error(`Search request failed with HTTP ${response.status}.`);
      }

      const payload: unknown = await response.json();
      if (!isSearchResponse(payload)) {
        throw new Error("Search response did not match the expected format.");
      }
      if (currentRequestId !== requestId.current) {
        return;
      }
      setRequestState({ status: "result", query: cleanQuery, response: payload });
    } catch {
      if (controller.signal.aborted || currentRequestId !== requestId.current) {
        return;
      }
      setRequestState({ status: "error" });
    }
  }

  return (
    <>
      <form className="search-form" role="search" onSubmit={submitSearch}>
        <label className="sr-only" htmlFor="search-query">Search AI Radar</label>
        <input
          className="search-input"
          id="search-query"
          type="search"
          placeholder="Ask about AI agents, models, policy..."
          value={query}
          onChange={(event) => {
            const nextQuery = event.target.value;
            setQuery(nextQuery);
            requestId.current += 1;
            activeController.current?.abort();
            activeController.current = null;
            setRequestState(nextQuery.trim() ? { status: "typing" } : { status: "idle" });
          }}
        />
        <button className="search-submit" type="submit">
          {requestState.status === "loading" ? "Searching…" : "Search"}
        </button>
      </form>
      {requestState.status === "typing" ? (
        <p className="quiet-state search-note" role="status">
          Press Enter or select Search to check verified AI Radar sources.
        </p>
      ) : requestState.status === "empty_query" ? (
        <p className="quiet-state search-note" role="status">Enter an AI question or topic to search.</p>
      ) : requestState.status === "loading" ? (
        <p className="quiet-state search-note" role="status">Searching verified AI Radar sources…</p>
      ) : requestState.status === "error" ? (
        <p className="quiet-state search-note" role="alert">Unable to search AI Radar right now. Please try again.</p>
      ) : requestState.status === "result" ? (
        <div className="search-results" aria-live="polite">
          <p className="quiet-state search-note">Results for: {requestState.query}</p>
          {requestState.response.status === "answered" ? (
            <p className="quiet-state search-answer">{requestState.response.answer}</p>
          ) : (
            <p className="quiet-state search-result-message">
              {requestState.response.message}
            </p>
          )}
          {"sources" in requestState.response && (
            <ul className="search-sources">
              {requestState.response.sources.map((source, index) => (
                <li
                  className="search-source-row"
                  key={`${source.url ?? source.title}-${index}`}
                >
                  {source.url ? (
                    <a className="search-source-title" href={source.url}>{source.title}</a>
                  ) : (
                    <span className="search-source-title">{source.title}</span>
                  )}
                  <p className="search-source-meta">
                    {[source.source, source.published, source.category]
                      .filter(Boolean)
                      .join(" · ")}
                  </p>
                  {source.summary && (
                    <p className="quiet-state search-source-summary">{source.summary}</p>
                  )}
                  {source.why_it_matters && (
                    <p className="quiet-state search-source-summary">{source.why_it_matters}</p>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      ) : null}
    </>
  );
}

function RecencyArcFallback() {
  return (
    <svg viewBox="0 0 640 480" preserveAspectRatio="xMidYMid meet" aria-hidden="true" focusable="false">
      <line className="baseline" x1="8" y1="370" x2="571" y2="370" />
      <path className="arc arc-now" fill="none" d="M 176.9 370 A 112.6 112.6 0 0 1 402.1 370" />
      <path className="arc arc-day" fill="none" d="M 92.5 370 A 197 197 0 0 1 486.5 370" />
      <path className="arc arc-week" fill="none" d="M 8 370 A 281.5 281.5 0 0 1 571 370" />
      <text className="label-now" x="410.1" y="362">Now</text>
      <text className="label-day" x="494.5" y="362">24h</text>
      <text className="label-week" x="579" y="362">7d</text>
    </svg>
  );
}

const EvolvedArcsCanvas = dynamic(
  () => import("./lab/hero-object/EvolvedArcsCanvas").then((module) => module.EvolvedArcsCanvas),
  { ssr: false, loading: () => <RecencyArcFallback /> },
);

function RecencyVisual({ reducedMotion }: { reducedMotion: boolean }) {
  return (
    <motion.div className="hero-visual" id="hero-visual" role="img" aria-label="An abstract composition of fine arcs labeled Now, 24h, and 7d" {...revealMotionProps(reducedMotion)}>
      <EvolvedArcsCanvas
        interactive={false}
        objectScale={1.5}
        showLabels
        transparent
        pointerEvents="none"
        fallback={<RecencyArcFallback />}
      />
    </motion.div>
  );
}

export default function Home() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [activeSection, setActiveSection] = useState<string | null>(null);
  const [footerInView, setFooterInView] = useState(false);
  const reducedMotion = useHydratedReducedMotion();
  const morningBriefing = useBriefing("morning");
  const eveningBriefing = useBriefing("evening");

  useEffect(() => {
    const sections = ["latest", "briefings", "search"]
      .map((id) => document.getElementById(id))
      .filter((section): section is HTMLElement => section !== null);
    const observedEntries = new Map<Element, IntersectionObserverEntry>();
    const trackedSections = new Set<Element>(sections);
    const footer = document.querySelector(".site-footer");
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          observedEntries.set(entry.target, entry);
          if (entry.target === footer && entry.isIntersecting) {
            setFooterInView(true);
          }
        });
        const visibleEntries = [...observedEntries.values()]
          .filter((entry) => trackedSections.has(entry.target) && entry.isIntersecting && entry.intersectionRatio >= 0.1);
        const hashTarget = window.location.hash.slice(1);
        const hashEntry = visibleEntries.find((entry) => (entry.target as HTMLElement).id === hashTarget);
        const activeEntry = hashEntry ?? visibleEntries.sort((left, right) => {
            const viewportCenter = window.innerHeight / 2;
            const leftBounds = left.target.getBoundingClientRect();
            const rightBounds = right.target.getBoundingClientRect();
            const leftCenter = (Math.max(0, leftBounds.top) + Math.min(window.innerHeight, leftBounds.bottom)) / 2;
            const rightCenter = (Math.max(0, rightBounds.top) + Math.min(window.innerHeight, rightBounds.bottom)) / 2;
            return Math.abs(leftCenter - viewportCenter) - Math.abs(rightCenter - viewportCenter);
          })[0];
        setActiveSection(activeEntry ? (activeEntry.target as HTMLElement).id : null);
      },
      { threshold: Array.from({ length: 21 }, (_, index) => index / 20) },
    );

    sections.forEach((section) => observer.observe(section));
    if (footer) observer.observe(footer);
    return () => observer.disconnect();
  }, []);

  return (
    <>
      <a className="skip-link" href="#main-content">Skip to content</a>
      <header className="site-header" id="top">
        <div className="page-container header-container">
          <a className="wordmark" href="#top" aria-label="AI Radar home">AI Radar</a>
          <button
            className="menu-toggle"
            type="button"
            aria-label={menuOpen ? "Close menu" : "Open menu"}
            aria-expanded={menuOpen}
            aria-controls="primary-navigation"
            onClick={() => setMenuOpen(!menuOpen)}
          >
            <span aria-hidden="true">{menuOpen ? "×" : "≡"}</span>
          </button>
          <nav className={`nav-links${menuOpen ? " is-open" : ""}`} id="primary-navigation" aria-label="Primary navigation">
            <a className={activeSection === "latest" ? "is-active" : undefined} href="#latest" onClick={() => { setMenuOpen(false); setActiveSection("latest"); }}>Latest</a>
            <a className={activeSection === "briefings" ? "is-active" : undefined} href="#briefings" onClick={() => { setMenuOpen(false); setActiveSection("briefings"); }}>Briefings</a>
            <a className={activeSection === "search" ? "is-active" : undefined} href="#search" onClick={() => { setMenuOpen(false); setActiveSection("search"); }}>Search</a>
          </nav>
        </div>
      </header>

      <main id="main-content" tabIndex={-1}>
        <section className="page-container hero-composition" aria-labelledby="hero-title">
          <motion.div className="intro-copy" {...heroMotionProps(reducedMotion)}>
            <motion.p className="eyebrow" {...heroChildProps(reducedMotion, 0)}>AI Radar / Independent intelligence</motion.p>
            <motion.p className="intro-title" id="intro-title" {...heroChildProps(reducedMotion, 0)}>Welcome to the AI world</motion.p>
            <motion.p className="intro-support" {...heroChildProps(reducedMotion, 2)}>Explore what is changing across AI.</motion.p>
            <motion.h1 id="hero-title" {...heroChildProps(reducedMotion, 1)}>What is changing across AI right now, and why does it matter?</motion.h1>
          </motion.div>
          <RecencyVisual reducedMotion={reducedMotion} />
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
              morningState={morningBriefing}
              eveningState={eveningBriefing}
            />
          </section>

          {timeRanges.map((range) => (
            <section className="time-chapter" id={range.id} aria-labelledby={`${range.id}-title`} key={range.id}>
              <p className={`time-label time-label-${range.tone}`}>{range.label}</p>
              <h3 id={`${range.id}-title`}>{range.title}</h3>
              <StoryRows window={range.window} />
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
          </motion.article>
        </section>

        <motion.section className="page-container static-section" id="search" aria-labelledby="search-title" {...staggerMotionProps(reducedMotion)}>
          <div className="section-heading static-section-heading">
            <div className="static-heading-copy">
              <motion.p className="section-index" {...staggerChildProps(reducedMotion)}>04 / SEARCH</motion.p>
              <motion.h2 id="search-title" {...staggerChildProps(reducedMotion)}>Search AI Radar</motion.h2>
            </div>
          </div>
          <SearchPanel />
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
              requestState={morningBriefing}
            />
            <BriefingPanel
              kind="evening"
              title="Evening briefing"
              description="Compares new developments against the morning baseline."
              requestState={eveningBriefing}
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