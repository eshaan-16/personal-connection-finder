from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Signal categories, highest-trust first. Every source and candidate is tagged
# with one of these. The goal is a target's reachable NICHE circle — friends,
# schoolmates, and early-venture collaborators rank highest; family is kept but
# demoted (the user can find close family on their own).
SIGNAL_CATEGORIES = (
    "close_friend",                # close / longtime / childhood personal friend
    "school_tie",                  # classmate, roommate, dorm, fraternity, lab
    "early_venture",               # co-founder/early employee of a pre-fame venture
    "professional_co_occurrence",  # colleague, board member, advisor, business partner
    "institutional_affiliation",   # alumni, fellowship, cohort (broader than school_tie)
    "social_proof",                # mentor, "early investor", "first believed in"
    "family",                      # spouse, sibling, parent, adult child, relative
    "joint_appearance",            # shared panel, conference, podcast
    "incidental",                  # broad/incidental: yearbook, neighborhood, misc
)

# Relative trust of each category. Friends, school ties, and early ventures are
# exactly what this tool now exists to surface, so they rank highest; family is
# intentionally demoted below professional/school ties; incidental lowest.
SIGNAL_WEIGHT = {
    "close_friend": 1.00,
    "school_tie": 0.95,
    "early_venture": 0.93,
    "professional_co_occurrence": 0.85,
    "institutional_affiliation": 0.82,
    "social_proof": 0.78,
    "family": 0.70,
    "joint_appearance": 0.58,
    "incidental": 0.35,
}

CONFIDENCE_TIERS = ("high", "medium", "low")

# Public-prominence buckets the extractor assigns to each person, and the fame
# score each maps to (0 = private/unknown person, 1 = globally famous). Used to
# filter out well-known people so niche, harder-to-reach connections surface.
PROMINENCE_FAME = {
    "household_name": 1.00,   # globally famous — most people would recognize them
    "industry_known": 0.65,   # well known in their field / has own Wikipedia page
    "niche": 0.30,            # some public footprint, not widely known
    "private": 0.10,          # ordinary person, minimal public profile
}
_PROMINENCE_RANK = {"household_name": 3, "industry_known": 2, "niche": 1, "private": 0}
# Unrated defaults to "niche" so a missing rating never over-filters a real lead.
_DEFAULT_FAME = PROMINENCE_FAME["niche"]


def fame_from_prominence(prominence: str) -> float:
    return PROMINENCE_FAME.get((prominence or "").strip().lower(), _DEFAULT_FAME)


def strongest_prominence(values) -> str:
    """Return the most-prominent bucket among ``values`` (used when merging the
    same person seen across several sources)."""
    best, best_rank = "", -1
    for value in values:
        v = (value or "").strip().lower()
        rank = _PROMINENCE_RANK.get(v, -1)
        if rank > best_rank:
            best, best_rank = v, rank
    return best


@dataclass
class SearchResult:
    """One web result from one provider for one query."""
    query: str
    signal_category: str
    provider: str
    title: str
    url: str
    snippet: str = ""
    rank: int = 0
    published_date: str = ""  # ISO 8601 (YYYY-MM-DD) when known, else ""
    page_text: str = ""


@dataclass
class Source:
    """A single citation supporting a candidate<->target connection."""
    url: str
    domain: str = ""
    title: str = ""
    snippet: str = ""
    quote: str = ""
    published_date: str = ""
    provider: str = ""
    query: str = ""
    signal_category: str = "incidental"

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "domain": self.domain,
            "title": self.title,
            "snippet": self.snippet,
            "quote": self.quote,
            "published_date": self.published_date,
            "provider": self.provider,
            "query": self.query,
            "signal_category": self.signal_category,
        }


@dataclass
class Candidate:
    """A named individual who co-occurs with the target, with all citations."""
    name: str
    name_key: str
    explanation: str = ""
    signal_categories: set[str] = field(default_factory=set)
    sources: list[Source] = field(default_factory=list)
    extraction_confidence: float = 0.0  # max confidence reported by extractor
    prominence: str = ""  # strongest public-prominence bucket seen for this person

    def add_source(self, source: Source) -> None:
        # Deduplicate by URL within a candidate.
        for existing in self.sources:
            if existing.url.rstrip("/") == source.url.rstrip("/"):
                return
        self.sources.append(source)
        if source.signal_category:
            self.signal_categories.add(source.signal_category)


@dataclass
class ScoredCandidate:
    candidate: Candidate
    score: float = 0.0
    tier: str = "low"
    in_network: bool = False
    degree: Optional[int] = None  # 1 = direct, 2 = connection-of-connection
    via: str = ""                 # which 1st-degree contact bridges, if degree 2
    approximate_match: bool = False  # network match was first+last only — verify
    most_recent_date: str = ""
    oldest_date: str = ""
    distinct_domains: int = 0
    stale_only: bool = False      # only evidence is >STALE_YEARS old
    rationale: str = ""
    fame: float = 0.0             # 0 = private/unknown, 1 = globally famous
    prominence: str = ""          # the prominence bucket driving `fame`

    def to_dict(self) -> dict:
        cand = self.candidate
        return {
            "name": cand.name,
            "score": round(self.score, 4),
            "tier": self.tier,
            "explanation": cand.explanation,
            "fame": round(self.fame, 3),
            "prominence": self.prominence,
            "signal_categories": sorted(cand.signal_categories),
            "primary_signal": best_signal(cand.signal_categories),
            "in_network": self.in_network,
            "degree": self.degree,
            "via": self.via,
            "approximate_match": self.approximate_match,
            "distinct_domains": self.distinct_domains,
            "most_recent_date": self.most_recent_date,
            "oldest_date": self.oldest_date,
            "stale_only": self.stale_only,
            "rationale": self.rationale,
            "sources": [s.to_dict() for s in cand.sources],
        }


def best_signal(categories: set[str]) -> str:
    """Highest-trust category present, used as the candidate's primary signal."""
    if not categories:
        return "incidental"
    return max(categories, key=lambda c: SIGNAL_WEIGHT.get(c, 0.0))
