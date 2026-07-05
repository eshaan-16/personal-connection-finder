from __future__ import annotations

import math
import re
from typing import Optional

from .dates import years_old
from .extract import RawCandidate
from .models import (
    SIGNAL_WEIGHT,
    Candidate,
    ScoredCandidate,
    SearchResult,
    Source,
    best_signal,
    fame_from_prominence,
    strongest_prominence,
)
from .network import NetworkIndex, NetworkMatch
from .util import normalize_name, registrable_domain, squeeze

# Weights of the four scoring terms (sum to 1.0 before boosts/penalties).
W_SIGNAL = 0.34
W_CORROBORATION = 0.26
W_RECENCY = 0.22
W_CONFIDENCE = 0.18

NETWORK_BOOST = {1: 0.12, 2: 0.06}
STALE_PENALTY = 0.45
# Nudge niche/unknown people up and famous people down, so the ranking favours
# the harder-to-reach connections the user actually wants.
OBSCURITY_BONUS = 0.10

TIER_HIGH = 0.62
TIER_MEDIUM = 0.42


def _better_explanation(current: str, candidate_raw: RawCandidate, current_method: str) -> tuple[str, str]:
    """Prefer an LLM-written, more specific explanation over a generic one."""
    new = squeeze(candidate_raw.explanation)
    if not current:
        return new, candidate_raw.method
    # Gemini explanations beat heuristic ones; otherwise keep the longer line.
    if candidate_raw.method == "gemini" and current_method != "gemini":
        return new, "gemini"
    if candidate_raw.method == current_method and len(new) > len(current) + 8:
        return new, current_method
    return current, current_method


def merge_candidates(
    raws: list[RawCandidate],
    results_by_url: dict[str, SearchResult],
) -> list[Candidate]:
    by_key: dict[str, Candidate] = {}
    explanation_method: dict[str, str] = {}
    for raw in raws:
        key = normalize_name(raw.name)
        if not key:
            continue
        candidate = by_key.get(key)
        if candidate is None:
            candidate = Candidate(name=raw.name, name_key=key)
            by_key[key] = candidate
            explanation_method[key] = ""

        result = results_by_url.get(raw.citation_url)
        published = raw.published_date or (result.published_date if result else "")
        source = Source(
            url=raw.citation_url,
            domain=registrable_domain(raw.citation_url),
            title=result.title if result else "",
            snippet=result.snippet if result else raw.evidence_quote,
            quote=raw.evidence_quote,
            published_date=published,
            provider=result.provider if result else "",
            query=result.query if result else "",
            signal_category=raw.signal_category,
        )
        candidate.add_source(source)
        candidate.extraction_confidence = max(candidate.extraction_confidence, raw.confidence)
        candidate.prominence = strongest_prominence([candidate.prominence, raw.prominence])
        explanation, method = _better_explanation(
            candidate.explanation, raw, explanation_method[key]
        )
        candidate.explanation = explanation
        explanation_method[key] = method
    consolidated = _consolidate_surnames(list(by_key.values()))
    return _consolidate_subsets(consolidated)


def _fold_into(src: Candidate, dst: Candidate) -> None:
    for source in src.sources:
        dst.add_source(source)
    dst.extraction_confidence = max(dst.extraction_confidence, src.extraction_confidence)
    dst.prominence = strongest_prominence([dst.prominence, src.prominence])
    if src.explanation and not dst.explanation:
        dst.explanation = src.explanation


def _consolidate_surnames(candidates: list[Candidate]) -> list[Candidate]:
    """Fold a bare single-token candidate ("Nadella", "Phoebe") into the unique
    full-name candidate it clearly belongs to.

    A bare token folds into a multi-word candidate only when that candidate is
    the ONLY one whose first OR last token equals the bare token — i.e. there is
    exactly one person it could refer to. If the token matches more than one
    person (a common surname like "Gates", or a first name shared by two
    candidates like "Mary"), it is ambiguous and left untouched, so two
    different people are never conflated.
    """
    by_first: dict[str, list[Candidate]] = {}
    by_surname: dict[str, list[Candidate]] = {}
    for cand in candidates:
        tokens = cand.name_key.split()
        if len(tokens) >= 2:
            by_first.setdefault(tokens[0], []).append(cand)
            by_surname.setdefault(tokens[-1], []).append(cand)

    kept: list[Candidate] = []
    for cand in candidates:
        tokens = cand.name_key.split()
        if len(tokens) == 1:
            token = tokens[0]
            # Every full-name candidate this bare token could denote.
            targets = {id(c): c for c in by_first.get(token, []) + by_surname.get(token, [])}
            if len(targets) == 1:
                _fold_into(cand, next(iter(targets.values())))
                continue  # drop the bare duplicate
        kept.append(cand)
    return kept


_CONNECTOR_TOKENS = {"and", "with"}


def _is_conjunction(name_key: str) -> bool:
    """True if the name joins multiple people ("Bill and Melinda Gates")."""
    return bool(set(name_key.split()) & _CONNECTOR_TOKENS)


def _content_tokens(name_key: str) -> frozenset:
    return frozenset(t for t in name_key.split() if t not in _CONNECTOR_TOKENS)


def _consolidate_subsets(candidates: list[Candidate]) -> list[Candidate]:
    """Merge middle-name variants of the SAME person — "Melinda Gates" ->
    "Melinda French Gates", "Rory Gates" -> "Rory John Gates", "Paul Allen" ->
    "Paul Gardner Allen", "Mary Gates" -> "Mary Maxwell Gates".

    A candidate X folds into Y ONLY when it is unambiguous that they are the
    same individual:
      - X's tokens are a strict subset of Y's,
      - X and Y agree on BOTH first name and surname (so only middle tokens
        differ — a genuine name expansion, not two different people),
      - Y is not a multi-person conjunction, and
      - Y is the unique candidate meeting all of the above.

    This deliberately does NOT merge two incomparable names that merely share a
    spurious superset (e.g. "Michael Jordan" + "Michael Jackson" + a bogus
    "Michael Jordan Jackson"), nor absorb individuals into a conjunction
    ("Bill Gates" + "Melinda Gates" + "Bill and Melinda Gates") — both of which
    would fuse or drop distinct people.
    """
    entries = []
    for cand in candidates:
        key = cand.name_key
        toks = key.split()
        entries.append({
            "cand": cand,
            "key": key,
            "set": _content_tokens(key),
            "first": toks[0] if toks else "",
            "last": toks[-1] if toks else "",
            "conj": _is_conjunction(key),
        })
    n = len(entries)
    absorbed: dict[int, int] = {}

    for i in range(n):
        e_i = entries[i]
        if not e_i["set"] or e_i["conj"]:
            continue
        supers = [
            j for j in range(n)
            if j != i and not entries[j]["conj"]
            and e_i["set"] < entries[j]["set"]
            and e_i["first"] == entries[j]["first"]
            and e_i["last"] == entries[j]["last"]
        ]
        if len(supers) == 1:
            absorbed[i] = supers[0]

    def root(idx: int) -> int:
        seen: set[int] = set()
        while idx in absorbed and idx not in seen:
            seen.add(idx)
            idx = absorbed[idx]
        return idx

    for i in list(absorbed):
        r = root(i)
        if r == i:
            continue
        _fold_into(entries[i]["cand"], entries[r]["cand"])

    return [entries[i]["cand"] for i in range(n) if i not in absorbed]


def _recency_factor(most_recent: Optional[float], stale_years: int, recent_years: int) -> float:
    if most_recent is None:
        return 0.5  # unknown date is neutral, not penalized
    if most_recent <= recent_years:
        return 1.0 - 0.2 * (most_recent / max(recent_years, 1))
    if most_recent <= stale_years:
        span = max(stale_years - recent_years, 1)
        return 0.8 - 0.6 * ((most_recent - recent_years) / span)
    return 0.12


def _corroboration_factor(distinct_domains: int) -> float:
    if distinct_domains <= 0:
        return 0.0
    # 1 domain -> ~0.43, 2 -> ~0.68, 3 -> ~0.86, 4+ -> 1.0
    return min(1.0, math.log2(1 + distinct_domains) / math.log2(5))


_WIKI_SLUG_RE = re.compile(r"/wiki/([^?#]+)")


def _has_own_wikipedia(candidate: Candidate) -> bool:
    """True if any source is the person's OWN Wikipedia article — a strong,
    free signal that they are a notable public figure. (Merely being cited on
    the target's Wikipedia page does not count.)"""
    name_toks = set(candidate.name_key.split())
    if not name_toks:
        return False
    for source in candidate.sources:
        if "wikipedia.org" not in (registrable_domain(source.url) or ""):
            continue
        match = _WIKI_SLUG_RE.search(source.url)
        if not match:
            continue
        slug_toks = set(normalize_name(match.group(1).replace("_", " ")).split())
        # The article title must actually be this person (their name tokens are
        # a subset of the article title's tokens).
        if slug_toks and name_toks.issubset(slug_toks):
            return True
    return False


def _fame_score(candidate: Candidate, distinct_domains: int) -> float:
    """Estimate how well-known a person is, 0 (private) .. 1 (globally famous).

    Primary signal is the extractor's prominence bucket; a dedicated Wikipedia
    article and very broad multi-domain coverage bump it up. No extra API calls.
    """
    fame = fame_from_prominence(candidate.prominence)
    if _has_own_wikipedia(candidate):
        fame = max(fame, 0.75)  # own Wikipedia article => at least industry-known
    if distinct_domains >= 6:
        fame = min(1.0, fame + 0.10)  # covered everywhere => famous
    return max(0.0, min(1.0, fame))


def score_candidate(
    candidate: Candidate,
    *,
    network: Optional[NetworkIndex],
    stale_years: int,
    recent_years: int,
) -> ScoredCandidate:
    dates = [s.published_date for s in candidate.sources if s.published_date]
    most_recent_iso = max(dates) if dates else ""
    oldest_iso = min(dates) if dates else ""
    most_recent_age = years_old(most_recent_iso) if most_recent_iso else None

    distinct_domains = len({s.domain for s in candidate.sources if s.domain})
    signal_base = SIGNAL_WEIGHT.get(best_signal(candidate.signal_categories), 0.3)
    corroboration = _corroboration_factor(distinct_domains)
    recency = _recency_factor(most_recent_age, stale_years, recent_years)
    confidence = max(0.0, min(1.0, candidate.extraction_confidence))

    base = (
        W_SIGNAL * signal_base
        + W_CORROBORATION * corroboration
        + W_RECENCY * recency
        + W_CONFIDENCE * confidence
    )

    # Stale: the most recent evidence is older than the stale window, with no
    # recent corroboration to redeem it.
    stale_only = most_recent_age is not None and most_recent_age > stale_years
    if stale_only:
        base *= STALE_PENALTY

    match = network.lookup(candidate.name) if network else NetworkMatch()
    boost = NETWORK_BOOST.get(match.degree or 0, 0.0)
    if match.approximate:
        boost *= 0.5  # name-only match — don't fully trust it

    # Favour niche / hard-to-reach people: obscurity nudges the score up, fame
    # nudges it down, so well-known people rank lower even when kept.
    fame = _fame_score(candidate, distinct_domains)
    obscurity_adj = OBSCURITY_BONUS * (1.0 - 2.0 * fame)  # +bonus at fame 0, -bonus at fame 1
    score = max(0.0, min(1.0, base + boost + obscurity_adj))

    if score >= TIER_HIGH:
        tier = "high"
    elif score >= TIER_MEDIUM:
        tier = "medium"
    else:
        tier = "low"

    rationale = _rationale(
        signal=best_signal(candidate.signal_categories),
        distinct_domains=distinct_domains,
        recency_age=most_recent_age,
        stale_only=stale_only,
        match=match,
        prominence=candidate.prominence,
    )

    return ScoredCandidate(
        candidate=candidate,
        score=score,
        tier=tier,
        in_network=match.in_network,
        degree=match.degree,
        via=match.via,
        approximate_match=match.approximate,
        most_recent_date=most_recent_iso,
        oldest_date=oldest_iso,
        distinct_domains=distinct_domains,
        stale_only=stale_only,
        rationale=rationale,
        fame=fame,
        prominence=candidate.prominence,
    )


def _rationale(*, signal, distinct_domains, recency_age, stale_only, match: NetworkMatch,
               prominence: str = "") -> str:
    parts = [signal.replace("_", " ")]
    if prominence:
        parts.append(prominence.replace("_", " "))
    if distinct_domains >= 2:
        parts.append(f"{distinct_domains} independent sources")
    else:
        parts.append("single source")
    if recency_age is None:
        parts.append("undated")
    elif recency_age <= 5:
        parts.append("recent")
    elif stale_only:
        parts.append(f"stale (~{int(recency_age)}y old, penalized)")
    else:
        parts.append(f"~{int(recency_age)}y old")
    if match.in_network:
        hedge = " (name match — verify)" if match.approximate else ""
        if match.degree == 1:
            parts.append(f"already a 1st-degree connection{hedge}")
        elif match.degree == 2:
            parts.append(f"2nd-degree via {match.via}{hedge}")
    return "; ".join(parts)


def score_and_rank(
    raws: list[RawCandidate],
    results_by_url: dict[str, SearchResult],
    *,
    network: Optional[NetworkIndex] = None,
    stale_years: int = 15,
    recent_years: int = 5,
    remove_famous: bool = False,
    max_fame: float = 0.6,
    min_results: int = 0,
) -> tuple[list[ScoredCandidate], list[ScoredCandidate]]:
    """Return (kept, removed_famous).

    When ``remove_famous`` is set, anyone whose fame score is >= ``max_fame`` is
    pulled out of the main list (they're easy to find on your own) and returned
    separately. But ``min_results`` is a FLOOR: if fame filtering leaves fewer
    than ``min_results`` people, the least-famous of the removed are added back
    (closest to the niche threshold first) until the floor is met — so the user
    still gets a full list rather than a handful. Both lists are sorted best-first.
    """
    candidates = merge_candidates(raws, results_by_url)
    scored = [
        score_candidate(c, network=network, stale_years=stale_years, recent_years=recent_years)
        for c in candidates
    ]

    def rank_key(s: ScoredCandidate):
        return (s.score, s.distinct_domains, s.most_recent_date)

    kept, removed = scored, []
    if remove_famous:
        kept = [s for s in scored if s.fame < max_fame]
        removed = [s for s in scored if s.fame >= max_fame]

        # Floor: backfill from the removed (least-famous, then best-scoring first)
        # so the user always gets a usable number of leads.
        if min_results and len(kept) < min_results and removed:
            backfill = sorted(removed, key=lambda s: (s.fame, -s.score))
            need = min_results - len(kept)
            promoted = backfill[:need]
            promoted_ids = {id(s) for s in promoted}
            kept = kept + promoted
            removed = [s for s in removed if id(s) not in promoted_ids]

    for group in (kept, removed):
        group.sort(key=rank_key, reverse=True)
    return kept, removed
