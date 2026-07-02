from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_PLACEHOLDER_HINTS = ("your_", "_here", "changeme", "xxxx")


def _clean(value: str) -> str:
    value = (value or "").strip().strip('"').strip("'")
    if not value:
        return ""
    if value.lower() in {"none", "null"}:
        return ""
    lowered = value.lower()
    if any(hint in lowered for hint in _PLACEHOLDER_HINTS):
        return ""  # treat obvious .env placeholders as "not set"
    return value


def load_env(path: str = ".env") -> None:
    """Populate os.environ from a .env file without overriding real env vars."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Settings:
    # --- Search providers (any subset may be configured) ---
    brave_api_key: str = ""
    google_cse_id: str = ""
    google_cse_key: str = ""
    bing_api_key: str = ""

    # --- Extraction LLM (optional; falls back to heuristic if absent) ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # --- Tunables ---
    max_results_per_query: int = 6
    max_pages_per_query: int = 3   # how many results to fetch full text for
    max_queries: int = 0           # 0 = use the full generated batch
    fetch_pages: bool = True
    analyze_photos: bool = False   # vision-analyze uncaptioned images (needs Gemini)
    max_photos: int = 4            # cap vision calls per run (cost control)
    request_delay: float = 0.2     # polite delay between outbound web fetches
    http_retries: int = 4
    allow_insecure_ssl: bool = False
    stale_years: int = 15          # sources older than this with no recent
                                   # corroboration get penalized
    recent_years: int = 5          # window that counts as "recent" corroboration

    # --- Persistence ---
    db_path: str = "connection_finder.sqlite3"
    cache_ttl_hours: int = 168     # reuse cached search results for a week
    use_cache: bool = True

    # --- Local network index (optional) ---
    connections_csv: str = ""
    second_degree_json: str = ""

    # Providers the user explicitly restricted to (empty = all available).
    only_providers: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        load_env()
        env = os.environ.get
        settings = cls(
            # BRAVE_API_KEY is the spec name; BRAVE_SEARCH_API_KEY is the
            # artemisv1 name. Accept either.
            brave_api_key=_clean(env("BRAVE_API_KEY", "") or env("BRAVE_SEARCH_API_KEY", "")),
            google_cse_id=_clean(env("GOOGLE_CSE_ID", "")),
            google_cse_key=_clean(env("GOOGLE_CSE_KEY", "")),
            bing_api_key=_clean(env("BING_API_KEY", "")),
            gemini_api_key=_clean(env("GEMINI_API_KEY", "")),
            gemini_model=_clean(env("GEMINI_MODEL", "")) or "gemini-2.5-flash",
        )
        for key, value in overrides.items():
            if value is not None and hasattr(settings, key):
                setattr(settings, key, value)
        return settings

    # --- Capability checks (drive graceful degradation) ---
    def has_brave(self) -> bool:
        return bool(self.brave_api_key)

    def has_google_cse(self) -> bool:
        return bool(self.google_cse_id and self.google_cse_key)

    def has_bing(self) -> bool:
        return bool(self.bing_api_key)

    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    def available_providers(self) -> list[str]:
        names = []
        if self.has_brave():
            names.append("brave")
        if self.has_google_cse():
            names.append("google_cse")
        if self.has_bing():
            names.append("bing")
        if self.only_providers:
            wanted = {p.strip().lower() for p in self.only_providers}
            names = [n for n in names if n in wanted]
        return names

    def validate_for_search(self) -> None:
        if not self.available_providers():
            raise ConfigError(
                "No search provider is configured. Set at least one of:\n"
                "  BRAVE_API_KEY        (https://brave.com/search/api/)\n"
                "  GOOGLE_CSE_ID + GOOGLE_CSE_KEY  (https://programmablesearchengine.google.com/)\n"
                "  BING_API_KEY         (Azure Bing Web Search; note Microsoft is retiring this API)\n"
                "in your environment or a .env file. See .env.example."
            )


class ConfigError(RuntimeError):
    pass
