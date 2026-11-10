"use client";

import { usePathname } from "next/navigation";
import { useEffect, useState, type KeyboardEvent, type MouseEvent } from "react";

const homeSections = [
  { id: "latest", label: "Latest", icon: "latest" },
  { id: "research", label: "Research", icon: "research" },
  { id: "models", label: "Models", icon: "models" },
  { id: "agents", label: "Agents", icon: "agents" },
  { id: "search", label: "Search", icon: "search" },
  { id: "briefings", label: "Briefings", icon: "briefings" },
] as const;

type SectionIconName = (typeof homeSections)[number]["icon"];

function SectionIcon({ name }: { name: SectionIconName }) {
  const commonProps = {
    "aria-hidden": true as const,
    fill: "none",
    height: 18,
    stroke: "currentColor",
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    strokeWidth: 1.6,
    viewBox: "0 0 24 24",
    width: 18,
  };

  switch (name) {
    case "latest":
      return <svg {...commonProps}><path d="M5 4.5h14v15H5z" /><path d="M8 8h8M8 12h8M8 16h5" /></svg>;
    case "research":
      return <svg {...commonProps}><path d="M9 3h6M10 3v6l-5.5 8.7A2.2 2.2 0 0 0 6.4 21h11.2a2.2 2.2 0 0 0 1.9-3.3L14 9V3" /><path d="M8 15h8" /></svg>;
    case "models":
      return <svg {...commonProps}><path d="m12 3 8 4.5-8 4.5-8-4.5L12 3Z" /><path d="m4 12 8 4.5 8-4.5M4 16.5 12 21l8-4.5" /></svg>;
    case "agents":
      return <svg {...commonProps}><rect x="5" y="7" width="14" height="12" rx="3" /><path d="M12 4v3M9 12h.01M15 12h.01M9 16h6M3 11v4M21 11v4" /></svg>;
    case "search":
      return <svg {...commonProps}><circle cx="10.8" cy="10.8" r="6.3" /><path d="m16 16 4.2 4.2" /></svg>;
    case "briefings":
      return <svg {...commonProps}><path d="M6 3.5h8l4 4V20H6z" /><path d="M14 3.5V8h4M9 12h6M9 16h6" /></svg>;
  }
}

function RadarMark() {
  return (
    <svg
      aria-hidden="true"
      className="section-navigation-mark"
      fill="none"
      height="24"
      viewBox="0 0 24 24"
      width="24"
    >
      <circle cx="12" cy="12" r="8.5" stroke="currentColor" strokeWidth="1.5" />
      <path d="M12 12 18.1 6.2M12 3.5v2M3.5 12h2M12 18.5v2M18.5 12h2" stroke="currentColor" strokeLinecap="round" strokeWidth="1.5" />
      <circle cx="12" cy="12" r="1.6" fill="currentColor" />
    </svg>
  );
}

export function SectionNavigation() {
  const pathname = usePathname();
  const [visibleSection, setVisibleSection] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState(false);
  const activeSection =
    pathname === "/search" || pathname === "/briefings"
      ? pathname.slice(1)
      : pathname === "/"
        ? visibleSection
        : null;

  useEffect(() => {
    if (pathname !== "/") {
      return;
    }

    const sections = homeSections
      .map(({ id }) => document.getElementById(id))
      .filter((section): section is HTMLElement => section !== null);
    const updateActiveSection = () => {
      const activationLine = Math.min(window.innerHeight * 0.2, 180);
      const isAtPageEnd =
        window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2;
      const active =
        (isAtPageEnd
          ? sections.at(-1)
          : sections
              .map((section) => ({ section, top: section.getBoundingClientRect().top }))
              .filter(({ top }) => top <= activationLine)
              .at(-1)?.section) ?? sections[0];
      setVisibleSection(active?.id ?? null);
    };
    let scrollFrame = 0;
    const observer = new IntersectionObserver(
      updateActiveSection,
      { rootMargin: "-20% 0px -35% 0px", threshold: 0 },
    );

    sections.forEach((section) => observer.observe(section));
    const handleScroll = () => {
      if (scrollFrame) return;
      scrollFrame = window.requestAnimationFrame(() => {
        scrollFrame = 0;
        updateActiveSection();
      });
    };
    window.addEventListener("scroll", handleScroll, { passive: true });
    window.addEventListener("resize", handleScroll);
    updateActiveSection();
    return () => {
      observer.disconnect();
      window.removeEventListener("scroll", handleScroll);
      window.removeEventListener("resize", handleScroll);
      if (scrollFrame) window.cancelAnimationFrame(scrollFrame);
    };
  }, [pathname]);

  const isHome = pathname === "/";
  const toggleLabel = collapsed ? "Expand sidebar" : "Collapse sidebar";
  const toggleSidebar = () => setCollapsed((isCollapsed) => !isCollapsed);

  function handleSidebarClick(event: MouseEvent<HTMLElement>) {
    const target = event.target;
    if (!(target instanceof Element)) {
      return;
    }
    if (target.closest("a, button, input, select, textarea, [role='button'], [contenteditable='true']")) {
      return;
    }
    if (window.getSelection()?.toString()) {
      return;
    }
    toggleSidebar();
  }

  function handleSidebarKeyDown(event: KeyboardEvent<HTMLElement>) {
    if (event.target !== event.currentTarget || (event.key !== "Enter" && event.key !== " ")) {
      return;
    }
    event.preventDefault();
    toggleSidebar();
  }

  return (
    <nav
      className={`section-navigation${collapsed ? " is-collapsed" : ""}`}
      aria-label={`Page sections, ${collapsed ? "collapsed" : "expanded"}`}
      aria-controls="section-navigation-links"
      tabIndex={0}
      onClick={handleSidebarClick}
      onKeyDown={handleSidebarKeyDown}
    >
      <div className="section-navigation-header">
        <button
          className="section-navigation-brand"
          type="button"
          aria-label={`${toggleLabel} using AI Radar`}
          aria-expanded={!collapsed}
          aria-controls="section-navigation-links"
          title={toggleLabel}
          onClick={toggleSidebar}
        >
          <RadarMark />
          <span className="section-navigation-brand-name">AI Radar</span>
        </button>
      </div>
      <ul id="section-navigation-links">
        {homeSections.map(({ id, label, icon }) => {
          const href =
            id === "search" && !isHome
              ? "/search"
              : id === "briefings" && !isHome
                ? "/briefings"
                : isHome
                  ? `#${id}`
                  : `/#${id}`;
          const isCurrent = activeSection === id;

          return (
            <li key={id}>
              <a
                className={isCurrent ? "is-active" : undefined}
                href={href}
                aria-label={label}
                aria-current={isCurrent ? "location" : undefined}
                data-tooltip={label}
                title={label}
              >
                <SectionIcon name={icon} />
                <span className="section-navigation-label">{label}</span>
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
