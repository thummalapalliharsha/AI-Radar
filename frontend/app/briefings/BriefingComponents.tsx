"use client";

import { useEffect, useState } from "react";

const apiBaseUrl = (
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"
).replace(/\/+$/, "");

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

export type BriefingRequestState =
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

export function useBriefing(kind: "morning" | "evening") {
  const [requestState, setRequestState] = useState<BriefingRequestState>({
    status: "loading",
  });
  const [attempt, setAttempt] = useState(0);

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
  }, [kind, attempt]);

  return {
    requestState,
    retry: () => {
      setRequestState({ status: "loading" });
      setAttempt((currentAttempt) => currentAttempt + 1);
    },
  };
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

export function SinceMorningStatus({
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

export function BriefingPanel({
  kind,
  title,
  description,
  requestState,
  onRetry,
  headingLevel = "h3",
}: {
  kind: "morning" | "evening";
  title: string;
  description: string;
  requestState: BriefingRequestState;
  onRetry?: () => void;
  headingLevel?: "h2" | "h3";
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
  const Heading = headingLevel;

  return (
    <article className={`briefing-block briefing-block-${kind}`}>
      <Heading>{title}</Heading>
      <p className="quiet-state">{description}</p>
      <div className="briefing-divider" aria-hidden="true" />
      {requestState.status === "loaded" ? (
        <>
          <p className="briefing-state" role="status">{currentStateLabel}</p>
          {kind === "evening" && requestState.briefing.baseline_missing && (
            <p className="quiet-state briefing-empty-state" role="status">
              The morning baseline is missing. The stories below are shown in the order supplied by the briefing and cannot be compared with a morning snapshot.
            </p>
          )}
          <p className="briefing-summary quiet-state">
            {requestState.briefing.summary}
          </p>
          <p className="briefing-generated">
            Generated{" "}
            <time dateTime={requestState.briefing.generated_at}>
              {formatBriefingTimestamp(requestState.briefing.generated_at)}
            </time>
            {" · "}
            {requestState.briefing.generated_local_date}
          </p>
          {kind === "evening" && requestState.briefing.baseline_local_date && (
            <p className="briefing-generated">
              Morning baseline date: {requestState.briefing.baseline_local_date}
            </p>
          )}
          {requestState.briefing.stories.length > 0 ? (
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
                          aria-label={`Read ${story.title} from ${story.source || "the original source"} (opens in a new tab)`}
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
        <>
          <p
            className={kind === "morning" ? "awaiting-tag" : "quiet-state briefing-empty-state"}
            role={requestState.status === "error" ? "alert" : "status"}
          >
            {stateMessage[requestState.status]}
          </p>
          {requestState.status === "error" && onRetry && (
            <button className="search-submit briefing-retry" type="button" onClick={onRetry}>
              Retry {kind} briefing
            </button>
          )}
        </>
      )}
    </article>
  );
}
