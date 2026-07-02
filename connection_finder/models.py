from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Signal categories, highest-trust first. Every source and candidate is tagged
# with one of these. The goal is CLOSE personal connections (family and friends
# rank highest), then professional/institutional ties, then incidental mentions.
SIGNAL_CATEGORIES = (
    "family",                      # spouse, sibling, parent, adult child, relative
    "close_friend",                # close / longtime / childhood personal friend
    "professional_co_occurrence",  # co-founder, board member, advisor, colleague
    "institutional_affiliation",   # alumni, fellowship, classmate, lab
    "social_proof",                # mentor, "early investor", "first believed in"
    "joint_appearance",            # shared panel, conference, podcast
    "incidental",                  # broad/incidental: yearbook, neighborhood, misc
)

# Relative trust of each category. Family + close friends are exactly what this
# tool exists to surface, so they rank highest; incidental mentions lowest.
SIGNAL_WEIGHT = {
    "family": 1.00,
    "close_friend": 0.95,
    "professional_co_occurrence": 0.90,
    "institutional_affiliation": 0.78,
    "social_proof": 0.72,
    "joint_appearance": 0.58,
    "incidental": 0.30,
}

CONFIDENCE_TIERS = ("high", "medium", "low")


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

    def to_dict(self) -> dict:
        cand = self.candidate
        return {
            "name": cand.name,
            "score": round(self.score, 4),
            "tier": self.tier,
            "explanation": cand.explanation,
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
