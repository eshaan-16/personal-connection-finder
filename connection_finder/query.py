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

    # 1. Family — the closest ties, highest priority.
    add(f"{head} wife OR husband OR spouse", "family")
    add(f"{head} brother OR sister OR sibling", "family")
    add(f"{head} son OR daughter OR children", "family")
    add(f"{head} father OR mother OR parents", "family")
    add(f"{head} family OR relatives", "family")
    add(f"{head} married OR wedding", "family")

    # 2. Close friends.
    add(f'{head} "close friend" OR "best friend"', "close_friend")
    add(f'{head} "longtime friend" OR "childhood friend"', "close_friend")
    add(f"{head} friend OR friendship", "close_friend")

    # 3. Niche / social sites — where personal ties surface most.
    add(f"{head} site:facebook.com", "close_friend")
    add(f"{head} site:instagram.com", "close_friend")
    add(f"{head} site:twitter.com OR site:x.com", "close_friend")

    # 4. Close professional co-occurrence (co-founders/partners are often close).
    add(f"{head} co-founder OR cofounder", "professional_co_occurrence")
    add(f"{head} colleague OR partner", "professional_co_occurrence")
    add(f"{head} board member OR advisor", "professional_co_occurrence")

    # 5. Institutional affiliation.
    add(f"{head} classmate OR alumni OR roommate", "institutional_affiliation")

    # 6. Public social proof.
    add(f'{head} mentor OR "early investor"', "social_proof")

    # 7. Joint appearances.
    add(f"{head} panel OR conference OR podcast", "joint_appearance")

    # 8. Broad / incidental — the disambiguated name alone, catches the rest.
    add(head, "incidental")

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
