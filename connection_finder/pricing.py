from __future__ import annotations

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
