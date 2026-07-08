from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class QuerySpec:
    text: str
    signal_category: str


def _q(name: str) -> str:
    return f'"{name}"'


def build_queries(
    target_name: str,
    context: str,
    *,
    location: Optional[str] = None,
    industry: Optional[str] = None,
    period: Optional[str] = None,
    limit: int = 0,
) -> list[QuerySpec]:
    """Generate the disambiguated query batch aimed at the target's CLOSE circle.

    Priority is family and close friends, then close professional/institutional
    ties, then broad probes — including niche and social sites (Facebook,
    Instagram, X). The target name and context are always quoted together so
    same-name false positives are filtered out and every result stays anchored
    to the right person (this is the main defence against irrelevant hits).
    """
    name = _q(target_name)
    ctx = _q(context) if context else ""
    head = f"{name} {ctx}".strip()

    specs: list[QuerySpec] = []

    def add(text: str, category: str) -> None:
        specs.append(QuerySpec(text=" ".join(text.split()), signal_category=category))

    # The query set is biased toward FRIENDS, SCHOOL TIES, and EARLY VENTURES —
    # the reachable niche circle — not family or famous associates (family is
    # demoted; celebrities get filtered downstream). No bare-name catch-all.

    # 1. Close friends and personal circle (highest priority).
    add(f'{head} "close friend" OR "childhood friend" OR "longtime friend" OR "best friend"', "close_friend")
    add(f'{head} "best man" OR "maid of honor" OR godfather OR godmother OR confidant', "close_friend")
    add(f'{head} neighbor OR "grew up with" OR "old friend"', "close_friend")

    # 2. School ties — classmates, roommates, dorm/fraternity, school friends.
    add(f'{head} roommate OR classmate OR dormmate OR "college friend"', "school_tie")
    add(f'{head} fraternity OR sorority OR "high school" OR yearbook', "school_tie")
    add(f'{head} "went to school with" OR schoolmate OR "lab partner" OR "study group"', "school_tie")

    # 3. Early ventures — co-founders/early employees of pre-fame startups.
    add(f'{head} "early employee" OR "co-founder" OR "first hire" OR "founding team"', "early_venture")
    add(f'{head} "first company" OR "first startup" OR "early venture" OR "before he was"', "early_venture")
    add(f'{head} "started" OR "launched" OR "early days" OR "in the beginning"', "early_venture")

    # 4. Deep WORK history — former colleagues, early jobs, direct reports.
    add(f'{head} "worked with" OR "worked alongside" OR "former colleague"', "professional_co_occurrence")
    add(f'{head} "early career" OR "first job" OR "hired by" OR "reported to"', "professional_co_occurrence")
    add(f'{head} assistant OR "chief of staff" OR aide OR mentor OR protege', "professional_co_occurrence")

    # 5. Niche / social sites — where lesser-known personal ties surface.
    add(f"{head} site:facebook.com", "close_friend")
    add(f"{head} site:instagram.com OR site:linkedin.com", "professional_co_occurrence")

    # 6. Social proof and personal events (surface the circle, not the family).
    add(f'{head} "early supporter" OR "believed in" OR "first backer"', "social_proof")
    add(f"{head} reunion OR memorial OR tribute OR alumni", "joint_appearance")

    # 7. Family — kept but minimal (demoted; user wants friends over family).
    add(f"{head} family OR sibling OR spouse OR relatives", "family")

    # Optional narrowing terms get their own incidental probes plus refine the
    # broad query toward the requested era/scene.
    extras = [term for term in (location, industry, period) if term]
    if extras:
        joined = " ".join(_q(term) if " " in term else term for term in extras)
        add(f"{head} {joined}", "incidental")
        if period:
            add(f"{head} {period}", "incidental")

    # Dedupe while preserving category-priority order.
    seen: set[str] = set()
    ordered: list[QuerySpec] = []
    for spec in specs:
        key = spec.text.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(spec)

    if limit and limit > 0:
        return ordered[:limit]
    return ordered
