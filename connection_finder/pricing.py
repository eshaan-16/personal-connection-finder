from __future__ import annotations

import math
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Approximate published rates (USD). These are ESTIMATES for planning only —
# update them if a provider changes pricing. Actual billing is authoritative.
# --------------------------------------------------------------------------- #

# Gemini generateContent: (input $ / 1M tokens, output $ / 1M tokens).
GEMINI_PRICES = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
}
_DEFAULT_GEMINI_PRICE = (0.30, 2.50)

# Search APIs: cost per query call (many have a free monthly/daily tier first).
SEARCH_PRICE_PER_CALL = {
    "brave": 3.0 / 1000,        # ~$3 per 1,000 (Base tier); 2,000/mo often free
    "google_cse": 5.0 / 1000,   # $5 per 1,000 after 100 free/day
    "bing": 15.0 / 1000,        # legacy Azure pricing; varies
}
_DEFAULT_SEARCH_PRICE = 3.0 / 1000


def gemini_model_price(model: str) -> tuple[float, float]:
    key = (model or "").strip().lower()
    if key in GEMINI_PRICES:
        return GEMINI_PRICES[key]
    # Fall back on a family prefix match (e.g. "gemini-2.5-flash-002").
    for name, price in GEMINI_PRICES.items():
        if key.startswith(name):
            return price
    return _DEFAULT_GEMINI_PRICE


@dataclass
class CostEstimate:
    model: str = ""
    gemini_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    gemini_cost: float = 0.0
    search_calls_by_provider: dict = field(default_factory=dict)
    search_cost: float = 0.0

    @property
    def search_calls(self) -> int:
        return sum(self.search_calls_by_provider.values())

    @property
    def total_cost(self) -> float:
        return self.gemini_cost + self.search_cost

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "gemini_calls": self.gemini_calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "gemini_cost_usd": round(self.gemini_cost, 4),
            "search_calls": self.search_calls,
            "search_calls_by_provider": self.search_calls_by_provider,
            "search_cost_usd": round(self.search_cost, 4),
            "total_cost_usd": round(self.total_cost, 4),
        }


# Cost/quality modes surfaced in the UI. "economy" swaps the extractor for the
# far cheaper flash-lite model (~5x less); "accurate" uses flash.
MODE_MODELS = {
    "economy": "gemini-2.5-flash-lite",
    "balanced": "gemini-2.5-flash",
    "accurate": "gemini-2.5-flash",
}

# Rough per-evidence-item token costs, calibrated from real runs (focused text +
# JSON output). Used only for the PRE-run estimate that drives the slider.
_TOK_INPUT_PER_PAGE = 500
_TOK_INPUT_PER_CALL = 750
_TOK_OUTPUT_PER_PAGE = 770


@dataclass
class RunPlan:
    """A pre-run estimate: how much evidence a target number of connections needs
    and what that will roughly cost. Powers the UI's cost slider."""
    target_connections: int
    mode: str
    model: str
    max_pages_total: int
    min_results: int
    max_results_per_query: int
    n_queries: int
    gemini_calls: int
    est_prompt_tokens: int
    est_output_tokens: int
    est_gemini_cost: float
    est_search_calls: int
    est_search_cost: float

    @property
    def est_total_cost(self) -> float:
        return self.est_gemini_cost + self.est_search_cost

    def as_dict(self) -> dict:
        return {
            "target_connections": self.target_connections,
            "mode": self.mode,
            "model": self.model,
            "max_pages_total": self.max_pages_total,
            "min_results": self.min_results,
            "max_results_per_query": self.max_results_per_query,
            "n_queries": self.n_queries,
            "gemini_calls": self.gemini_calls,
            "est_prompt_tokens": self.est_prompt_tokens,
            "est_output_tokens": self.est_output_tokens,
            "est_gemini_cost_usd": round(self.est_gemini_cost, 4),
            "est_search_calls": self.est_search_calls,
            "est_search_cost_usd": round(self.est_search_cost, 4),
            "est_total_cost_usd": round(self.est_total_cost, 4),
        }


def pages_for_connections(target_connections: int) -> int:
    """Evidence pages needed to yield ~target_connections after dedup/filtering
    (~2 pages per surviving niche connection, since many are famous and filtered),
    clamped to a sane range."""
    return int(min(70, max(10, round(max(1, target_connections) * 2.0))))


def plan_run(
    target_connections: int,
    *,
    mode: str = "balanced",
    n_queries: int = 20,
    batch_size: int = 8,
    max_results_per_query: int = 6,
    provider: str = "brave",
    cached_fraction: float = 0.0,
    assume_free_search: bool = True,
) -> RunPlan:
    """Estimate settings + cost for a run targeting ``target_connections``.

    ``cached_fraction`` (0..1) discounts the Gemini estimate for the share of
    batches expected to hit the extraction cache (a re-run is ~free).
    ``assume_free_search`` treats search as $0 (Brave's free monthly tier covers
    typical usage), so the estimate reflects the real marginal cost: Gemini."""
    mode = mode if mode in MODE_MODELS else "balanced"
    model = MODE_MODELS[mode]
    pages = pages_for_connections(target_connections)
    calls = math.ceil(pages / max(1, batch_size))
    cache_keep = max(0.0, min(1.0, 1.0 - cached_fraction))

    prompt_tokens = int((pages * _TOK_INPUT_PER_PAGE + calls * _TOK_INPUT_PER_CALL) * cache_keep)
    output_tokens = int(pages * _TOK_OUTPUT_PER_PAGE * cache_keep)
    inp_rate, out_rate = gemini_model_price(model)
    gemini_cost = prompt_tokens / 1_000_000 * inp_rate + output_tokens / 1_000_000 * out_rate

    search_price = 0.0 if assume_free_search else SEARCH_PRICE_PER_CALL.get(provider, _DEFAULT_SEARCH_PRICE)
    search_cost = n_queries * search_price

    return RunPlan(
        target_connections=int(target_connections),
        mode=mode,
        model=model,
        max_pages_total=pages,
        min_results=int(target_connections),
        max_results_per_query=max_results_per_query,
        n_queries=n_queries,
        gemini_calls=int(calls * cache_keep) if cache_keep else 0,
        est_prompt_tokens=prompt_tokens,
        est_output_tokens=output_tokens,
        est_gemini_cost=gemini_cost,
        est_search_calls=n_queries,
        est_search_cost=search_cost,
    )


def estimate_cost(
    *,
    model: str,
    gemini_calls: int,
    prompt_tokens: int,
    output_tokens: int,
    search_calls_by_provider: dict,
) -> CostEstimate:
    inp_rate, out_rate = gemini_model_price(model)
    gemini_cost = prompt_tokens / 1_000_000 * inp_rate + output_tokens / 1_000_000 * out_rate
    search_cost = sum(
        SEARCH_PRICE_PER_CALL.get(name, _DEFAULT_SEARCH_PRICE) * n
        for name, n in search_calls_by_provider.items()
    )
    return CostEstimate(
        model=model,
        gemini_calls=gemini_calls,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        gemini_cost=gemini_cost,
        search_calls_by_provider=dict(search_calls_by_provider),
        search_cost=search_cost,
    )
