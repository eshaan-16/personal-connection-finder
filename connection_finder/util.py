from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse

# Tokens that strongly indicate the "name" is actually an organization, place,
# section header, or boilerplate rather than a person. Used by the heuristic
# extractor and by name normalization.
ORG_STOPWORDS = {
    "inc", "llc", "ltd", "corp", "corporation", "company", "co", "group",
    "university", "college", "school", "institute", "institutes", "academy",
    "foundation", "fund", "capital", "ventures", "venture", "partners",
    "partner", "associates", "holdings", "labs", "lab", "press", "news",
    "times", "journal", "magazine", "review", "post", "today", "media",
    "podcast", "show", "conference", "summit", "forum", "council", "board",
    "committee", "department", "division", "office", "center", "centre",
    "league", "association", "society", "club", "team", "city", "county",
    "state", "street", "avenue", "road", "north", "south", "east", "west",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "united", "states", "america", "american", "national", "international",
    "global", "world", "linkedin", "facebook", "instagram", "twitter",
    "youtube", "google", "github", "crunchbase", "wikipedia",
}

# Common honorifics / suffixes stripped before comparing names.
NAME_AFFIXES = {
    "mr", "mrs", "ms", "miss", "dr", "prof", "professor", "sir", "dame",
    "jr", "sr", "ii", "iii", "iv", "phd", "md", "esq", "mba",
}

_WS = re.compile(r"\s+")
_NON_NAME = re.compile(r"[^a-z0-9 ]+")


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_name(name: str) -> str:
    """Lowercase, accent-fold, drop honorifics/suffixes and punctuation.

    Produces a stable key used to dedupe people across sources and to match a
    candidate against the local network index.
    """
    if not name:
        return ""
    value = strip_accents(name).lower()
    value = value.replace("&", " and ")
    value = _NON_NAME.sub(" ", value)
    tokens = [tok for tok in value.split() if tok and tok not in NAME_AFFIXES]
    return _WS.sub(" ", " ".join(tokens)).strip()


def name_tokens(name: str) -> list[str]:
    return [tok for tok in normalize_name(name).split() if tok]


def looks_like_person_name(name: str) -> bool:
    """Cheap filter: 2-3 capitalized tokens, none of which are org/stop words."""
    raw = name.strip()
    if not raw:
        return False
    tokens = raw.split()
    if not (2 <= len(tokens) <= 3):
        return False
    for token in tokens:
        bare = re.sub(r"[^A-Za-z]", "", token)
        if len(bare) < 2:
            return False
        if bare.lower() in ORG_STOPWORDS:
            return False
        # Require a leading capital (allows McX, O'X, hyphenated).
        if not token[:1].isupper():
            return False
        if token.isupper() and len(token) > 3:
            return False  # SHOUTING headers are not names
    return True


def domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return ""
    host = host.rsplit("@", 1)[-1]   # drop any userinfo
    host = host.split(":", 1)[0]     # drop port so :443 variants collapse
    if host.startswith("www."):
        host = host[4:]
    return host


def registrable_domain(url: str) -> str:
    """Best-effort eTLD+1 so 'a.blog.medium.com' and 'medium.com' count as one
    independent source. No PSL dependency; good enough for corroboration."""
    host = domain_of(url)
    if not host:
        return ""
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    # Handle common two-label public suffixes (co.uk, com.au, ...).
    two_label_suffixes = {
        "co.uk", "org.uk", "ac.uk", "gov.uk", "co.jp", "com.au", "org.au",
        "co.in", "co.nz", "com.br", "com.cn",
    }
    last_two = ".".join(parts[-2:])
    if last_two in two_label_suffixes:
        return ".".join(parts[-3:])
    return last_two


def squeeze(text: str, limit: int | None = None) -> str:
    cleaned = _WS.sub(" ", text or "").strip()
    if limit is not None and len(cleaned) > limit:
        return cleaned[:limit].rstrip() + "…"
    return cleaned
