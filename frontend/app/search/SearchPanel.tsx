"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";

const apiBaseUrl = (
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"
).replace(/\/+$/, "");

const maxQueryLength = 1000;

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

type SearchError =
  | { kind: "validation"; message: string }
  | { kind: "unavailable"; message: string }
  | { kind: "server"; message: string }
  | { kind: "network"; message: string }
  | { kind: "unexpected"; message: string };

type SearchRequestState =
  | { status: "idle" }
  | { status: "typing" }
  | { status: "empty_query" }
  | { status: "loading"; query: string }
  | { status: "error"; query: string; error: SearchError }
  | { status: "result"; query: string; response: SearchResponse };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isSearchError(value: unknown): value is SearchError {
  return (
    isRecord(value) &&
    (value.kind === "validation" ||
      value.kind === "unavailable" ||
      value.kind === "server" ||
      value.kind === "network" ||
      value.kind === "unexpected") &&
    typeof value.message === "string"
  );
}

function isSearchSource(value: unknown): value is SearchSource {
  return (
    isRecord(value) &&
    typeof value.title === "string" &&
    typeof value.source === "string" &&
    typeof value.category === "string" &&
    typeof value.published === "string" &&
    typeof value.summary === "string" &&
    typeof value.why_it_matters === "string" &&
    (typeof value.url === "string" || value.url === null)
  );
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
    value.sources.every(isSearchSource)
  );
}

function safeSourceUrl(value: string | null): string | null {
  if (!value) {
    return null;
  }
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:" ? url.href : null;
  } catch {
    return null;
  }
}

function getSearchError(status: number): SearchError {
  if (status === 422) {
    return {
      kind: "validation",
      message: "That query could not be accepted. Check it and try again.",
    };
  }
  if (status === 503) {
    return {
      kind: "unavailable",
      message: "Local AI search is currently unavailable. Try again in a moment.",
    };
  }
  if (status === 500) {
    return {
      kind: "server",
      message: "AI Radar could not complete the search. Please try again.",
    };
  }
  return {
    kind: "server",
    message: `The search request failed (HTTP ${status}). Please try again.`,
  };
}

function sourceHeading(source: SearchSource): string {
  return source.title || source.source || "Untitled source";
}

export function SearchPanel({
  headingLevel = "h2",
}: {
  headingLevel?: "h2" | "h3";
}) {
  const [query, setQuery] = useState("");
  const [requestState, setRequestState] = useState<SearchRequestState>({
    status: "idle",
  });
  const requestId = useRef(0);
  const activeController = useRef<AbortController | null>(null);
  const lastSubmittedQuery = useRef("");
  const Heading = headingLevel;

  useEffect(
    () => () => {
      requestId.current += 1;
      activeController.current?.abort();
    },
    [],
  );

  async function runSearch(cleanQuery: string) {
    activeController.current?.abort();

    const currentRequestId = ++requestId.current;
    const controller = new AbortController();
    activeController.current = controller;
    lastSubmittedQuery.current = cleanQuery;
    setRequestState({ status: "loading", query: cleanQuery });

    try {
      const searchUrl = new URL(`${apiBaseUrl}/api/search`);
      searchUrl.searchParams.set("q", cleanQuery);
      const response = await fetch(searchUrl, { signal: controller.signal });
      if (!response.ok) {
        throw getSearchError(response.status);
      }

      const payload: unknown = await response.json();
      if (!isSearchResponse(payload)) {
        throw {
          kind: "unexpected",
          message: "AI Radar returned an unexpected search response. Please try again.",
        } satisfies SearchError;
      }
      if (currentRequestId !== requestId.current) {
        return;
      }
      setRequestState({ status: "result", query: cleanQuery, response: payload });
    } catch (error) {
      if (controller.signal.aborted || currentRequestId !== requestId.current) {
        return;
      }
      const searchError: SearchError = isSearchError(error)
        ? error
        : {
            kind: "network",
            message: "Could not connect to AI Radar search. Check the connection and try again.",
          };
      setRequestState({ status: "error", query: cleanQuery, error: searchError });
    }
  }

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
    if (cleanQuery.length > maxQueryLength) {
      requestId.current += 1;
      setRequestState({
        status: "error",
        query: cleanQuery,
        error: {
          kind: "validation",
          message: `Keep your question to ${maxQueryLength} characters or fewer.`,
        },
      });
      return;
    }

    await runSearch(cleanQuery);
  }

  function handleQueryChange(value: string) {
    setQuery(value);
    requestId.current += 1;
    activeController.current?.abort();
    activeController.current = null;
    setRequestState(value.trim() ? { status: "typing" } : { status: "idle" });
  }

  function retrySearch() {
    const retryQuery = lastSubmittedQuery.current;
    if (retryQuery) {
      void runSearch(retryQuery);
    }
  }

  const isLoading = requestState.status === "loading";

  return (
    <>
      <form className="search-form" role="search" onSubmit={submitSearch}>
        <label className="sr-only" htmlFor="search-query">Search AI Radar</label>
        <input
          className="search-input"
          id="search-query"
          type="search"
          maxLength={maxQueryLength}
          aria-describedby="search-query-guidance"
          placeholder="Ask about AI agents, models, policy..."
          value={query}
          onChange={(event) => handleQueryChange(event.target.value)}
        />
        <button className="search-submit" type="submit">
          {isLoading ? "Searching…" : "Search"}
        </button>
      </form>
      <p className="quiet-state search-note" id="search-query-guidance">
        Ask about AI Radar sources. Maximum {maxQueryLength} characters.
        {query.length > 0 && ` ${query.length}/${maxQueryLength}`}
      </p>
      {requestState.status === "typing" ? (
        <p className="quiet-state search-note" role="status">
          Press Enter or select Search to check verified AI Radar sources.
        </p>
      ) : requestState.status === "empty_query" ? (
        <p className="quiet-state search-note" role="alert">
          Enter an AI question or topic to search.
        </p>
      ) : requestState.status === "loading" ? (
        <p className="quiet-state search-note" role="status" aria-live="polite">
          Searching verified AI Radar sources…
        </p>
      ) : requestState.status === "error" ? (
        <div className="search-note" role="alert">
          <p className="quiet-state">{requestState.error.message}</p>
          {requestState.error.kind !== "validation" && (
            <button className="search-submit briefing-retry" type="button" onClick={retrySearch}>
              Retry search
            </button>
          )}
        </div>
      ) : requestState.status === "result" ? (
        <div className="search-results" aria-live="polite" aria-atomic="false">
          <p className="quiet-state search-note">Results for: {requestState.query}</p>
          {requestState.response.status === "answered" ? (
            <>
              <Heading className="search-answer-heading">Answer</Heading>
              <p className="quiet-state search-answer">{requestState.response.answer}</p>
            </>
          ) : (
            <p className="quiet-state search-result-message" role="status">
              {requestState.response.message}
            </p>
          )}
          {"sources" in requestState.response && (
            <section className="search-source-section" aria-label="Sources">
              <Heading className="search-sources-heading">Sources</Heading>
              {requestState.response.sources.length > 0 ? (
                <ol className="search-sources">
                  {requestState.response.sources.map((source, index) => {
                    const sourceUrl = safeSourceUrl(source.url);
                    const title = sourceHeading(source);
                    return (
                      <li
                        className="search-source-row"
                        key={`${source.url ?? title}-${index}`}
                      >
                        {sourceUrl ? (
                          <a
                            className="search-source-title"
                            href={sourceUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            aria-label={`Read ${title} from ${source.source || "the original source"} (opens in a new tab)`}
                          >
                            {title}
                          </a>
                        ) : (
                          <span className="search-source-title">{title}</span>
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
                    );
                  })}
                </ol>
              ) : (
                <p className="quiet-state search-result-message" role="status">
                  No source articles were returned.
                </p>
              )}
            </section>
          )}
        </div>
      ) : null}
    </>
  );
}
