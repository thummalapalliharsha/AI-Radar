"""Canonicalization helpers for source article URLs."""

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
    "ref",
    "source",
}


def normalize_url(url: str) -> str:
    """Remove tracking parameters, fragments, and non-root trailing slashes."""
    if not url:
        return ""

    url = url.strip()
    try:
        parsed = urlparse(url)
        filtered_queries = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.lower() not in TRACKING_PARAMS
        ]
        normalized = parsed._replace(
            fragment="",
            query=urlencode(filtered_queries),
        )
        clean_url = urlunparse(normalized)
        if clean_url.endswith("/") and len(parsed.path) > 1:
            clean_url = clean_url[:-1]
        return clean_url
    except Exception:
        return url


def url_identity_variants(url: str) -> tuple[str, ...]:
    """Return the normalized URL and its equivalent terminal-slash spelling."""
    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    variants = [normalized]
    if len(parsed.path) > 1 and not parsed.path.endswith("/"):
        variants.append(urlunparse(parsed._replace(path=f"{parsed.path}/")))
    return tuple(variants)
