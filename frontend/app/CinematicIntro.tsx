"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";

const headline = "Time reveals what matters";
const introSeenKey = "ai-radar-intro-seen";
const introDurationMs = 6000;
const reducedMotionDurationMs = 850;
const exitDurationMs = 420;
const statusStages = [
  "GATHERING SIGNALS",
  "COLLECTING INTELLIGENCE",
  "CONNECTING DISCOVERIES",
  "PREPARING YOUR RADAR",
];
let introVisibilitySnapshot: boolean | undefined;

function subscribeToIntroVisibility() {
  return () => {};
}

function shouldShowIntro() {
  if (introVisibilitySnapshot !== undefined) {
    return introVisibilitySnapshot;
  }

  try {
    introVisibilitySnapshot = window.sessionStorage.getItem(introSeenKey) !== "1";
  } catch (error) {
    console.error("Unable to read the intro session state.", error);
    introVisibilitySnapshot = true;
  }
  return introVisibilitySnapshot;
}

function subscribeToReducedMotion(onChange: () => void) {
  const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  mediaQuery.addEventListener("change", onChange);
  return () => mediaQuery.removeEventListener("change", onChange);
}

function getReducedMotionPreference() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function focusMainContent() {
  const main = document.querySelector<HTMLElement>("main");
  if (!main) {
    return;
  }
  if (!main.hasAttribute("tabindex")) {
    main.tabIndex = -1;
  }
  main.focus({ preventScroll: true });
}

function setPageInert(inert: boolean) {
  const pageRegions = [
    document.querySelector<HTMLElement>(".section-navigation"),
    document.querySelector<HTMLElement>("main"),
    document.querySelector<HTMLElement>(".site-footer"),
  ].filter((region): region is HTMLElement => region !== null);

  pageRegions.forEach((region) => { region.inert = inert; });
}

function Hourglass() {
  return (
    <svg
      className="intro-hourglass"
      viewBox="0 0 300 380"
      role="img"
      aria-label="An illuminated hourglass with amber sand flowing into its lower chamber"
    >
      <defs>
        <linearGradient id="intro-frame" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#FFF0C8" />
          <stop offset="0.18" stopColor="#B77836" />
          <stop offset="0.32" stopColor="#F7D994" />
          <stop offset="0.5" stopColor="#9A5E2E" />
          <stop offset="0.72" stopColor="#E7B85F" />
          <stop offset="0.86" stopColor="#FFF0C8" />
          <stop offset="1" stopColor="#8F572B" />
        </linearGradient>
        <linearGradient id="intro-glass" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#D9E8ED" stopOpacity="0.035" />
          <stop offset="0.12" stopColor="#FFF5DB" stopOpacity="0.23" />
          <stop offset="0.24" stopColor="#DCEAF0" stopOpacity="0.045" />
          <stop offset="0.74" stopColor="#DCEAF0" stopOpacity="0.035" />
          <stop offset="0.88" stopColor="#FFF3D5" stopOpacity="0.2" />
          <stop offset="1" stopColor="#C8792C" stopOpacity="0.045" />
        </linearGradient>
        <linearGradient id="intro-glass-edge" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#FFF8E4" stopOpacity="0.78" />
          <stop offset="0.18" stopColor="#D8E9EF" stopOpacity="0.38" />
          <stop offset="0.5" stopColor="#FFF9E8" stopOpacity="0.1" />
          <stop offset="0.82" stopColor="#F4D9A0" stopOpacity="0.34" />
          <stop offset="1" stopColor="#FFF0C8" stopOpacity="0.74" />
        </linearGradient>
        <linearGradient id="intro-sand-upper" x1="0" y1="0" x2="0.8" y2="1">
          <stop offset="0" stopColor="#FFF0BB" />
          <stop offset="0.36" stopColor="#E9B55D" />
          <stop offset="0.78" stopColor="#D18A3C" />
          <stop offset="1" stopColor="#9D5527" />
        </linearGradient>
        <linearGradient id="intro-sand-lower" x1="0" y1="0" x2="0.8" y2="1">
          <stop offset="0" stopColor="#FFE9A5" />
          <stop offset="0.38" stopColor="#E8AE55" />
          <stop offset="0.78" stopColor="#C97A32" />
          <stop offset="1" stopColor="#91491F" />
        </linearGradient>
        <linearGradient id="intro-sand-stream" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#FFF5D0" />
          <stop offset="0.48" stopColor="#F4C86B" />
          <stop offset="1" stopColor="#D78A38" />
        </linearGradient>
        <linearGradient id="intro-frame-highlight" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#FFF3C9" stopOpacity="0.08" />
          <stop offset="0.22" stopColor="#FFF3C9" stopOpacity="0.86" />
          <stop offset="0.52" stopColor="#FFF3C9" stopOpacity="0.16" />
          <stop offset="0.82" stopColor="#FFF3C9" stopOpacity="0.72" />
          <stop offset="1" stopColor="#FFF3C9" stopOpacity="0.05" />
        </linearGradient>
        <radialGradient id="intro-hourglass-light">
          <stop offset="0" stopColor="#F4D9A0" stopOpacity="0.3" />
          <stop offset="1" stopColor="#C8792C" stopOpacity="0" />
        </radialGradient>
        <pattern id="intro-sand-grain" width="11" height="9" patternUnits="userSpaceOnUse">
          <circle cx="1.4" cy="2.1" r="0.58" fill="#FFF2C4" fillOpacity="0.78" />
          <circle cx="7.9" cy="1.2" r="0.44" fill="#8F4D24" fillOpacity="0.62" />
          <circle cx="4.3" cy="7.4" r="0.68" fill="#FFD985" fillOpacity="0.76" />
          <circle cx="10.1" cy="5.7" r="0.5" fill="#FFF6D8" fillOpacity="0.72" />
          <circle cx="8.6" cy="8.5" r="0.36" fill="#A85B29" fillOpacity="0.56" />
        </pattern>
        <filter id="intro-hourglass-shadow" x="-60%" y="-40%" width="220%" height="190%">
          <feGaussianBlur stdDeviation="10" />
        </filter>
        <filter id="intro-hourglass-stream-glow" x="-100%" y="-10%" width="300%" height="120%">
          <feGaussianBlur stdDeviation="2.5" />
        </filter>
        <clipPath id="intro-upper-chamber">
          <path d="M80 48Q150 35 220 48C215 96 201 131 174 156C160 169 153 177 154 184Q150 189 146 184C147 177 140 169 126 156C99 131 85 96 80 48Z" />
        </clipPath>
        <clipPath id="intro-lower-chamber">
          <path d="M146 208Q150 203 154 208C153 216 160 224 174 237C201 262 215 297 220 332Q150 345 80 332C85 297 99 262 126 237C140 224 147 216 146 208Z" />
        </clipPath>
      </defs>

      <ellipse cx="150" cy="192" rx="134" ry="174" fill="url(#intro-hourglass-light)" />
      <ellipse
        className="intro-hourglass-shadow"
        cx="151"
        cy="357"
        rx="78"
        ry="10"
        filter="url(#intro-hourglass-shadow)"
      />

      <path
        className="intro-hourglass-frame"
        d="M62 28Q150 9 238 28L232 43Q150 26 68 43ZM68 337Q150 354 232 337L238 352Q150 371 62 352Z"
        fill="url(#intro-frame)"
      />
      <path
        d="M73 38C79 91 89 128 111 151C128 166 143 178 145 188Q150 196 145 204C143 216 128 230 111 245C89 268 79 302 73 343"
        fill="none"
        stroke="url(#intro-frame)"
        strokeWidth="7"
        strokeLinecap="round"
      />
      <path
        d="M227 38C221 91 211 128 189 151C172 166 157 178 155 188Q150 196 155 204C157 216 172 230 189 245C211 268 221 302 227 343"
        fill="none"
        stroke="url(#intro-frame)"
        strokeWidth="7"
        strokeLinecap="round"
      />

      <path
        d="M80 48Q150 35 220 48C215 96 201 131 174 156C160 169 153 177 154 184Q150 189 146 184C147 177 140 169 126 156C99 131 85 96 80 48Z"
        fill="url(#intro-glass)"
        stroke="url(#intro-glass-edge)"
        strokeWidth="1.8"
      />
      <path
        d="M146 208Q150 203 154 208C153 216 160 224 174 237C201 262 215 297 220 332Q150 345 80 332C85 297 99 262 126 237C140 224 147 216 146 208Z"
        fill="url(#intro-glass)"
        stroke="url(#intro-glass-edge)"
        strokeWidth="1.8"
      />
      <path
        d="M146 184Q150 189 154 184V208Q150 203 146 208Z"
        fill="url(#intro-glass)"
        stroke="#FFF0C8"
        strokeOpacity="0.55"
        strokeWidth="1"
      />
      <path
        d="M84 51Q150 39 216 51C211 98 197 130 171 153C158 165 150 174 150 183M150 209C150 218 159 226 171 238C197 261 211 293 216 329M84 329C89 294 103 261 129 238C141 226 150 218 150 209"
        fill="none"
        stroke="url(#intro-glass-edge)"
        strokeOpacity="0.55"
        strokeWidth="0.85"
      />

      <g clipPath="url(#intro-upper-chamber)">
        <path
          className="intro-hourglass-sand intro-hourglass-sand-upper"
          d="M82 51Q150 37 218 51C216 87 209 107 192 119C176 130 163 138 154 152Q150 158 146 152C137 138 124 130 108 119C91 107 84 87 82 51Z"
          fill="url(#intro-sand-upper)"
        />
        <path
          className="intro-hourglass-sand intro-hourglass-sand-upper"
          d="M82 51Q150 37 218 51C216 87 209 107 192 119C176 130 163 138 154 152Q150 158 146 152C137 138 124 130 108 119C91 107 84 87 82 51Z"
          fill="url(#intro-sand-grain)"
          opacity="0.58"
        />
        <path d="M91 55Q150 43 209 55C207 84 201 102 186 114Q167 128 153 149Q150 154 147 149Q133 128 114 114C99 102 93 84 91 55Z" fill="none" stroke="#FFF4D1" strokeOpacity="0.4" strokeWidth="1" />
      </g>
      <path
        className="intro-hourglass-stream-glow"
        d="M150 187C149.6 211 150.4 239 150 264"
        fill="none"
        stroke="#D99543"
        strokeWidth="6"
        strokeLinecap="round"
        filter="url(#intro-hourglass-stream-glow)"
      />
      <path
        className="intro-hourglass-stream"
        d="M150 187C149.6 211 150.4 239 150 264"
        fill="none"
        stroke="url(#intro-sand-stream)"
        strokeWidth="1.65"
        strokeLinecap="round"
      />
      <path
        className="intro-hourglass-stream-flow"
        d="M150 187C149.6 211 150.4 239 150 264"
        fill="none"
        stroke="#FFF0BD"
        strokeWidth="0.62"
        strokeLinecap="round"
      />
      <g clipPath="url(#intro-lower-chamber)">
        <path
          className="intro-hourglass-sand intro-hourglass-sand-lower"
          d="M150 263C153 267 155 270 158 275C171 288 189 300 204 313C213 321 217 328 219 333Q150 344 81 333C84 328 88 321 96 313C111 300 129 288 142 275C145 270 147 267 150 263Z"
          fill="url(#intro-sand-lower)"
        />
        <path
          className="intro-hourglass-sand intro-hourglass-sand-lower"
          d="M150 263C153 267 155 270 158 275C171 288 189 300 204 313C213 321 217 328 219 333Q150 344 81 333C84 328 88 321 96 313C111 300 129 288 142 275C145 270 147 267 150 263Z"
          fill="url(#intro-sand-grain)"
          opacity="0.7"
        />
        <path d="M137 280Q145 274 150 267Q155 274 163 280" fill="none" stroke="#FFF2C4" strokeOpacity="0.56" strokeWidth="1.1" />
        <path d="M104 320Q125 304 139 288M196 320Q175 304 161 288" fill="none" stroke="#FFD985" strokeOpacity="0.25" strokeWidth="0.9" />
      </g>
      <circle className="intro-hourglass-grain" cx="150" cy="190" r="2" fill="#FFF0BD" />
      <circle className="intro-hourglass-grain intro-hourglass-grain-mid" cx="149.2" cy="194" r="1.2" fill="#FFE3A0" />
      <circle className="intro-hourglass-grain intro-hourglass-grain-late" cx="150" cy="196" r="1.5" fill="#D99543" />
      <circle className="intro-hourglass-grain intro-hourglass-grain-final" cx="150.8" cy="199" r="0.9" fill="#FFF0BD" />

      <path
        d="M89 57C96 105 111 138 136 159C143 166 147 175 148 183M211 64C204 106 190 137 164 159M89 322C97 280 112 249 137 230M211 322C203 280 188 249 163 230"
        fill="none"
        stroke="white"
        strokeOpacity="0.34"
        strokeWidth="2.1"
        strokeLinecap="round"
      />
      <path
        d="M91 60C98 104 111 136 136 157M89 321C97 280 112 249 137 230"
        fill="none"
        stroke="white"
        strokeOpacity="0.42"
        strokeWidth="0.85"
        strokeLinecap="round"
      />
      <path d="M65 28Q150 12 235 28M65 352Q150 368 235 352" fill="none" stroke="url(#intro-frame-highlight)" strokeWidth="1.4" />
      <path d="M84 48Q150 37 216 48M84 332Q150 343 216 332" fill="none" stroke="#FFF0C8" strokeOpacity="0.72" strokeWidth="1" />
    </svg>
  );
}

export function CinematicIntro() {
  const firstVisit = useSyncExternalStore(
    subscribeToIntroVisibility,
    shouldShowIntro,
    () => true,
  );
  const [dismissed, setDismissed] = useState(false);
  const [exiting, setExiting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusStage, setStatusStage] = useState(0);
  const [previousStatusStage, setPreviousStatusStage] = useState<number | null>(null);
  const statusStageRef = useRef(0);
  const reducedMotion = useSyncExternalStore(
    subscribeToReducedMotion,
    getReducedMotionPreference,
    () => false,
  );
  const animationFrame = useRef<number | undefined>(undefined);
  const exitTimer = useRef<number | undefined>(undefined);
  const dialog = useRef<HTMLDivElement>(null);
  const wasVisible = useRef(false);
  const visible = firstVisit && !dismissed;

  useEffect(() => {
    if (!visible) {
      if (wasVisible.current) {
        wasVisible.current = false;
        focusMainContent();
      }
      return;
    }

    try {
      window.sessionStorage.setItem(introSeenKey, "1");
    } catch (error) {
      console.error("Unable to save the intro session state.", error);
    }
  }, [visible]);

  useEffect(() => {
    if (!visible) {
      if (wasVisible.current) {
        wasVisible.current = false;
        focusMainContent();
      }
      return;
    }

    wasVisible.current = true;
    const previousBodyOverflow = document.body.style.overflow;
    const pageRegions = [
      document.querySelector<HTMLElement>(".section-navigation"),
      document.querySelector<HTMLElement>("main"),
      document.querySelector<HTMLElement>(".site-footer"),
    ].filter((region): region is HTMLElement => region !== null);
    const previousInertState = pageRegions.map((region) => region.inert);
    document.body.style.overflow = "hidden";
    setPageInert(true);
    dialog.current?.focus({ preventScroll: true });

    const prefersReducedMotion = getReducedMotionPreference();
    const duration = prefersReducedMotion ? reducedMotionDurationMs : introDurationMs;
    let startedAt: number | undefined;
    const animateProgress = (timestamp: number) => {
      startedAt ??= timestamp;
      const elapsed = timestamp - startedAt;
      const nextProgress = Math.min(elapsed / duration, 1);
      setProgress(Math.floor(nextProgress * 100));
      const nextStatusStage = Math.min(Math.floor((elapsed / duration) * 4), 3);
      if (nextStatusStage !== statusStageRef.current) {
        setPreviousStatusStage(statusStageRef.current);
        statusStageRef.current = nextStatusStage;
        setStatusStage(nextStatusStage);
      }

      if (nextProgress < 1) {
        animationFrame.current = window.requestAnimationFrame(animateProgress);
        return;
      }

      setProgress(100);
      setExiting(true);
      exitTimer.current = window.setTimeout(() => {
        setDismissed(true);
      }, prefersReducedMotion ? 180 : exitDurationMs);
    };
    animationFrame.current = window.requestAnimationFrame(animateProgress);

    return () => {
      if (animationFrame.current !== undefined) {
        window.cancelAnimationFrame(animationFrame.current);
      }
      window.clearTimeout(exitTimer.current);
      pageRegions.forEach((region, index) => {
        region.inert = previousInertState[index];
      });
      document.body.style.overflow = previousBodyOverflow;
    };
  }, [visible]);

  if (!visible) {
    return null;
  }

  return (
    <div
      className={`cinematic-intro${exiting ? " is-exiting" : ""}${reducedMotion ? " is-reduced-motion" : ""}`}
      ref={dialog}
      role="dialog"
      aria-modal="true"
      aria-labelledby="cinematic-intro-title"
      tabIndex={-1}
    >
      <div className="cinematic-intro-atmosphere" aria-hidden="true" />
      <div className="cinematic-intro-content">
        <p className="cinematic-intro-eyebrow">AI RADAR <span aria-hidden="true">/</span> INDEPENDENT INTELLIGENCE</p>
        <h1 className="cinematic-intro-title" id="cinematic-intro-title" aria-label={headline}>
          <span aria-hidden="true">
            {Array.from(headline).map((character, index) => (
              <span
                className="cinematic-intro-letter"
                key={`${character}-${index}`}
                style={{ animationDelay: `${0.24 + index * 0.035}s` }}
              >
                {character === " " ? "\u00a0" : character}
              </span>
            ))}
          </span>
        </h1>
        <Hourglass />
        <div className="cinematic-intro-progress">
          <div
            className="cinematic-intro-progress-track"
            role="progressbar"
            aria-label="Intro animation progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={progress}
          >
            <span
              className="cinematic-intro-progress-fill"
              style={{ width: `${progress}%` }}
            />
          </div>
          <span className="cinematic-intro-progress-value" aria-hidden="true">{progress}%</span>
          <p className="cinematic-intro-status" aria-live="polite" aria-atomic="true">
            {previousStatusStage !== null && (
              <span className="cinematic-intro-status-label is-leaving" aria-hidden="true">
                {statusStages[previousStatusStage]}
              </span>
            )}
            <span className="cinematic-intro-status-label is-entering" key={statusStage}>
              {statusStages[statusStage]}
            </span>
          </p>
          <p className="cinematic-intro-caption">A moment to find the signal.</p>
        </div>
      </div>
    </div>
  );
}
