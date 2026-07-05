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

    # The query set is deliberately biased toward FAMILY and PERSONAL/NICHE ties
    # (childhood, hometown, staff, neighbours) rather than famous business
    # associates — the goal is harder-to-reach people, and celebrity co-founders
    # get filtered out downstream anyway. There is no bare-name catch-all query
    # (it just surfaced generic, famous results).

    # 1. Family — the closest ties, highest priority.
    add(f"{head} wife OR husband OR spouse", "family")
    add(f"{head} brother OR sister OR cousin OR nephew OR niece", "family")
    add(f'{head} son OR daughter OR "son-in-law" OR "daughter-in-law"', "family")
    add(f"{head} father OR mother OR parents", "family")
    add(f"{head} family OR relatives OR siblings", "family")

    # 2. Close friends and personal circle.
    add(f'{head} "close friend" OR "childhood friend" OR "longtime friend"', "close_friend")
    add(f'{head} neighbor OR "family friend"', "close_friend")

    # 3. Niche / social sites — where lesser-known personal ties surface.
    add(f"{head} site:facebook.com", "close_friend")
    add(f"{head} site:instagram.com", "close_friend")

    # 4. Personal history — childhood, hometown, school (surfaces obscure people).
    add(f'{head} childhood OR hometown OR "grew up"', "incidental")
    add(f'{head} roommate OR classmate OR "high school"', "institutional_affiliation")

    # 5. Deep WORK history — former colleagues, early jobs, direct reports. These
    #    surface real working relationships beyond the famous co-founders.
    add(f'{head} "worked with" OR "worked for" OR "worked alongside"', "professional_co_occurrence")
    add(f'{head} "former colleague" OR "former boss" OR "former employee"', "professional_co_occurrence")
    add(f'{head} "early career" OR "first job" OR "started his career" OR "started her career"', "professional_co_occurrence")
    add(f'{head} "hired by" OR "reported to" OR "worked under" OR "right-hand"', "professional_co_occurrence")

    # 6. Staff and everyday collaborators (assistants/aides are niche, reachable).
    add(f'{head} assistant OR "chief of staff" OR spokesperson OR aide OR secretary', "professional_co_occurrence")

    # 7. OLD / past personal relationships — former ties and reconnections.
    add(f'{head} "old friend" OR "used to" OR reconnected OR "back then"', "incidental")
    add(f'{head} former OR ex OR "years ago" OR "early days"', "incidental")

    # 8. Deep personal circle — the people at private milestones.
    add(f'{head} "best man" OR "maid of honor" OR godfather OR godmother', "close_friend")
    add(f'{head} confidant OR "inner circle" OR mentor OR protege', "close_friend")

    # 9. Public social proof and personal events (weddings/funerals name the circle).
    add(f'{head} "early supporter" OR "believed in" OR "first backer"', "social_proof")
    add(f"{head} wedding OR funeral OR reunion OR memorial", "joint_appearance")

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
