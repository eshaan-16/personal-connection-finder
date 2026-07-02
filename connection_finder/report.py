from __future__ import annotations

import json
from pathlib import Path

from .models import ScoredCandidate, best_signal
from .pipeline import RunResult

_TIER_LABEL = {"high": "HIGH", "medium": "MED ", "low": "LOW "}


def _network_label(item: ScoredCandidate) -> str:
    if not item.in_network:
        return "not in network"
    hedge = " (name match — verify)" if item.approximate_match else ""
    if item.degree == 1:
        base = "1st-degree (you know them)" if not item.approximate_match else "1st-degree?"
        return base + hedge
    if item.degree == 2:
        base = f"2nd-degree via {item.via}" if item.via else "2nd-degree"
        return base + hedge
    return "in network"


def render_console(result: RunResult, limit: int = 25) -> str:
    lines: list[str] = []
    lines.append(f"Target: {result.target}   Context: {result.context}")
    lines.append(
        f"Providers: {', '.join(result.providers) or 'none'} | extractor: {result.extractor} "
        f"| queries: {result.queries_run} | web results: {result.results_seen} "
        f"| candidates: {len(result.scored)}"
    )
    for warning in result.warnings:
        lines.append(f"  ! {warning}")
    cost = result.cost
    if cost is not None and (cost.gemini_calls or cost.search_calls):
        search_bits = ", ".join(f"{n} {p}" for p, n in cost.search_calls_by_provider.items())
        lines.append(
            f"Estimated cost this run: ~${cost.total_cost:0.4f} "
            f"(Gemini ${cost.gemini_cost:0.4f} for {cost.prompt_tokens:,}+{cost.output_tokens:,} tokens "
            f"over {cost.gemini_calls} calls; search ${cost.search_cost:0.4f}"
            + (f" — {search_bits}" if search_bits else "") + ")"
        )
        lines.append("  (estimate — see pricing.py; cached searches are free)")
    lines.append("")
    if not result.scored:
        lines.append("No candidate connectors were found. Try a broader context or add a provider key.")
        return "\n".join(lines)

    lines.append("Rank  Tier  Score  Network                       Candidate")
    lines.append("-" * 78)
    for rank, item in enumerate(result.scored[:limit], 1):
        net = _network_label(item)
        star = "*" if item.in_network else " "
        lines.append(
            f"{rank:>3}.  {_TIER_LABEL.get(item.tier, item.tier)}  {item.score:0.3f} "
            f"{star}{net:<28.28} {item.candidate.name}"
        )
        primary = best_signal(item.candidate.signal_categories).replace("_", " ")
        lines.append(f"        {primary}: {item.candidate.explanation}")
        lines.append(f"        why: {item.rationale}")
        for source in item.candidate.sources[:3]:
            date = source.published_date or "n.d."
            lines.append(f"        - [{date}] {source.url}")
        if len(item.candidate.sources) > 3:
            lines.append(f"        - (+{len(item.candidate.sources) - 3} more sources)")
        lines.append("")
    return "\n".join(lines)


def render_markdown(result: RunResult) -> str:
    lines = [
        f"# Warm-intro connectors for {result.target}",
        "",
        f"**Context:** {result.context}  ",
        f"**Providers:** {', '.join(result.providers) or 'none'} · "
        f"**Extractor:** {result.extractor} · "
        f"**Queries:** {result.queries_run} · **Candidates:** {len(result.scored)}",
        "",
        "_Every claim below is backed by a public-source citation. Treat these as "
        "possible public overlaps, not confirmed relationships, until you verify._",
        "",
    ]
    if result.warnings:
        lines.append("> Warnings: " + "; ".join(result.warnings))
        lines.append("")
    if not result.scored:
        lines.append("No candidate connectors were found.")
        return "\n".join(lines)

    for rank, item in enumerate(result.scored, 1):
        cand = item.candidate
        net = _network_label(item)
        lines.append(f"## {rank}. {cand.name}  —  {item.tier.upper()} ({item.score:0.3f})")
        lines.append("")
        lines.append(f"- **Connection:** {cand.explanation}")
        lines.append(f"- **Primary signal:** {best_signal(cand.signal_categories).replace('_', ' ')}")
        lines.append(f"- **In your network:** {net}")
        lines.append(f"- **Why this rank:** {item.rationale}")
        lines.append(f"- **Sources ({len(cand.sources)}):**")
        for source in cand.sources:
            date = source.published_date or "n.d."
            title = source.title or source.domain or source.url
            lines.append(f"    - [{date}] [{title}]({source.url}) — _{source.signal_category.replace('_', ' ')}_")
            if source.quote:
                lines.append(f"        > {source.quote}")
        lines.append("")
    return "\n".join(lines)


def write_markdown(result: RunResult, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(render_markdown(result), encoding="utf-8")


def write_json(result: RunResult, path: str) -> None:
    payload = {
        "target": result.target,
        "context": result.context,
        "providers": result.providers,
        "extractor": result.extractor,
        "queries_run": result.queries_run,
        "results_seen": result.results_seen,
        "warnings": result.warnings,
        "cost_estimate": result.cost.as_dict() if result.cost else None,
        "candidates": [item.to_dict() for item in result.scored],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
