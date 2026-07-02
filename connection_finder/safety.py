from __future__ import annotations

import re

# This tool's goal is to surface a target's CLOSE personal network — including
# family and friends — from PUBLIC sources, for warm introductions. So adult
# family relationships are ALLOWED (they are the point). What stays blocked is
# non-negotiable: minors, and private/contact/leaked/medical data. Those are
# never surfaced regardless of how "public" a page claims to be.
#
# Patterns are written to tolerate PLURALS and common phrasings — a filter that
# only matched singular forms ("teen" but not "teenagers", "year old" but not
# "years old") would let obvious minors through.

SENSITIVE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # --- Private / contact / identity data — never surfaced (plural-safe) ---
        r"\b(?:home|private|residential)\s+address(?:es)?\b",
        r"\bphone\s+numbers?\b",
        r"\bssns?\b", r"\bsocial\s+security\b", r"\bpassports?\b",
        r"\bleaked\b", r"\bdoxx?ed\b", r"\bdoxx?\b",
        r"\b(?:hidden|private)\s+accounts?\b",
        r"\bmedical\s+records?\b", r"\bhealth\s+conditions?\b",
        r"\barrest\s+records?\b",

        # --- Minor protection — blocked even though adult family is allowed ---
        r"\bunderage\b",
        # "minor child/children/son/daughter/sibling/boy/girl", and bare "minors".
        r"\bminor\s+(?:child(?:ren)?|sons?|daughters?|siblings?|boys?|girls?|kids?)\b",
        r"\bminors\b",
        # teen / teens / teenage / teenager(s) / teenaged.
        r"\bteen(?:agers?|aged|age|s)?\b",
        # school-age / grade-school signals.
        r"\bschool(?:boys?|girls?)\b",
        r"\b(?:kindergarten(?:er)?|preschool(?:er)?|middle\s+schooler|elementary\s+schooler)\b",
        r"\b(?:[1-9]|1[0-2])(?:st|nd|rd|th)\s+grader?\b",
        # young children (plural-safe). "baby" requires a following child noun
        # so the adult idiom "baby brother/sister", "baby of the family", and the
        # surname "Baby" don't wrongly drop a legitimate adult.
        r"\b(?:toddlers?|infants?|newborns?)\b",
        r"\bbab(?:y|ies)\s+(?:boys?|girls?|daughters?|sons?|twins?)\b",
        # Any stated age 0-17 implies a minor: "16-year-old", "8 years old",
        # "aged 12", "age 15".
        r"\b(?:[0-9]|1[0-7])[\s-]?years?[\s-]?old\b",
        r"\bage[ds]?\s+(?:[0-9]|1[0-7])\b",
    )
]


def is_sensitive(*texts: str) -> bool:
    """True if the combined text leans on private-data or minor signals.

    Adult family and friend relationships are intentionally NOT flagged here —
    surfacing them is the tool's purpose. Only genuinely off-limits content
    (private/contact/leaked/medical data, or any indication of a minor) is
    dropped.
    """
    blob = " ".join(t for t in texts if t)
    if not blob:
        return False
    return any(pattern.search(blob) for pattern in SENSITIVE_PATTERNS)
