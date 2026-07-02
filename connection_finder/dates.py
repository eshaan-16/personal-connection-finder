from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_SLASH = re.compile(r"\b(\d{4})/(\d{1,2})/(\d{1,2})\b")
_MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
# e.g. "May 3, 2018" / "3 May 2018"
_TEXT_DATE = re.compile(
    r"\b(\d{1,2}\s+)?([A-Za-z]{3,9})\.?\s+(\d{1,2})?,?\s*(\d{4})\b"
)
# Bare year as a last resort (1990..2099), e.g. URLs or "class of 2004".
_YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")


def _valid(year: int, month: int, day: int) -> Optional[str]:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def to_iso_date(value: str) -> str:
    """Best-effort coercion of an arbitrary date string to YYYY-MM-DD ("" if none)."""
    if not value:
        return ""
    value = value.strip()

    match = _ISO.search(value)
    if match:
        iso = _valid(int(match[1]), int(match[2]), int(match[3]))
        if iso:
            return iso

    match = _SLASH.search(value)
    if match:
        iso = _valid(int(match[1]), int(match[2]), int(match[3]))
        if iso:
            return iso

    match = _TEXT_DATE.search(value)
    if match:
        month = _MONTH_NAMES.get(match[2][:3].lower())
        if month:
            day = max(1, int(match[3] or match[1] or 1))
            iso = _valid(int(match[4]), month, day)
            if iso:
                return iso
            # Real but impossible day for the month (e.g. Feb 30) -> year only.
            return f"{int(match[4]):04d}-01-01"

    match = _YEAR.search(value)
    if match:
        return f"{match[1]}-01-01"
    return ""


def parse_iso(value: str) -> Optional[date]:
    if not value:
        return None
    match = _ISO.search(value)
    if not match:
        return None
    try:
        return date(int(match[1]), int(match[2]), int(match[3]))
    except ValueError:
        return None


def years_old(iso_value: str, today: Optional[date] = None) -> Optional[float]:
    parsed = parse_iso(iso_value)
    if not parsed:
        return None
    today = today or datetime.now().date()
    return max(0.0, (today - parsed).days / 365.25)


def extract_date_from_html(html_text: str, url: str = "") -> str:
    """Pull a publication date from meta tags / JSON-LD / visible text / URL."""
    if html_text:
        meta_patterns = (
            r'property=["\']article:published_time["\'][^>]*content=["\']([^"\']+)',
            r'content=["\']([^"\']+)["\'][^>]*property=["\']article:published_time["\']',
            r'name=["\'](?:date|pubdate|publishdate|dc\.date|sailthru\.date)["\'][^>]*content=["\']([^"\']+)',
            r'itemprop=["\']datePublished["\'][^>]*content=["\']([^"\']+)',
            r'"datePublished"\s*:\s*"([^"]+)"',
            r'<time[^>]*datetime=["\']([^"\']+)',
        )
        for pattern in meta_patterns:
            match = re.search(pattern, html_text, flags=re.IGNORECASE)
            if match:
                iso = to_iso_date(match.group(1))
                if iso:
                    return iso
    # Date embedded in the URL path, e.g. /2019/04/12/.
    url_match = re.search(r"/(20[0-4]\d|19[5-9]\d)/(\d{1,2})(?:/(\d{1,2}))?/", url)
    if url_match:
        iso = _valid(int(url_match[1]), int(url_match[2]), int(url_match[3] or 1))
        if iso:
            return iso
    return ""
