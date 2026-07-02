from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import zip_longest

from ..config import Settings
from ..http_util import AuthError, HttpError, RateLimitError
from ..models import SearchResult
from .base import SearchProvider
from .bing import BingProvider
from .brave import BraveProvider
from .google_cse import GoogleCseProvider


def build_providers(settings: Settings) -> list[SearchProvider]:
    """Instantiate exactly the providers whose keys are present (and allowed)."""
    available = set(settings.available_providers())
    providers: list[SearchProvider] = []
    if "brave" in available:
        providers.append(BraveProvider(
            settings.brave_api_key, retries=settings.http_retries,
            allow_insecure_ssl=settings.allow_insecure_ssl))
    if "google_cse" in available:
        providers.append(GoogleCseProvider(
            settings.google_cse_id, settings.google_cse_key,
            retries=settings.http_retries, allow_insecure_ssl=settings.allow_insecure_ssl))
    if "bing" in available:
        providers.append(BingProvider(
            settings.bing_api_key, retries=settings.http_retries,
            allow_insecure_ssl=settings.allow_insecure_ssl))
    return providers


def _result_key(url: str) -> str:
    return url.split("#", 1)[0].rstrip("/").lower()


def _merge(per_provider: list[list[SearchResult]]) -> list[SearchResult]:
    """Round-robin interleave provider result lists, dedupe by URL, and fold a
    missing publication date in from whichever provider supplied one."""
    merged: list[SearchResult] = []
    seen: dict[str, SearchResult] = {}
    for group in zip_longest(*per_provider):
        for result in group:
            if result is None:
                continue
            key = _result_key(result.url)
            if not key:
                continue
            if key in seen:
                kept = seen[key]
                if not kept.published_date and result.published_date:
                    kept.published_date = result.published_date
                if not kept.snippet and result.snippet:
                    kept.snippet = result.snippet
                continue
            seen[key] = result
            merged.append(result)
    return merged


class SearchEngine:
    """Single ``search(query)`` abstraction fanning out across all providers."""

    def __init__(self, providers: list[SearchProvider], *, cache=None, max_workers: int = 4,
                 verbose: bool = True):
        self.providers = providers
        self.cache = cache
        self.max_workers = max(1, min(max_workers, len(providers) or 1))
        self.verbose = verbose
        self._disabled: set[str] = set()  # providers rate-limited this run
        self.stats: dict[str, dict] = {p.name: {"ok": 0, "fail": 0, "errors": []} for p in providers}

    @property
    def provider_names(self) -> list[str]:
        return [p.name for p in self.providers]

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, file=sys.stderr)

    def _cache_key(self, query: str, count: int, active_names: list[str]) -> str:
        # Key on the providers that ACTUALLY ran, not every configured provider.
        # Otherwise a degraded run (a provider rate-limited mid-run) would cache
        # its partial result under the full-provider key and serve it to a later
        # healthy run as if complete. Keying on the active subset is self-healing:
        # a healthy run computes the full key, misses, and re-queries.
        sig = ",".join(sorted(active_names))
        return f"{sig}|{count}|{query}"

    def search(self, query: str, count: int, signal_category: str = "") -> list[SearchResult]:
        active = [p for p in self.providers if p.name not in self._disabled]
        if not active:
            return []
        active_names = [p.name for p in active]
        cache_key = self._cache_key(query, count, active_names)

        if self.cache is not None:
            cached = self.cache.get_search(cache_key)
            if cached is not None:
                for result in cached:
                    result.signal_category = signal_category
                return cached

        per_provider: list[list[SearchResult]] = []
        succeeded: list[str] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_provider = {
                pool.submit(provider.search, query, count): provider for provider in active
            }
            for future in as_completed(future_to_provider):
                provider = future_to_provider[future]
                try:
                    results = future.result()
                    per_provider.append(results)
                    succeeded.append(provider.name)
                    self.stats[provider.name]["ok"] += 1
                except RateLimitError as error:
                    self._disabled.add(provider.name)
                    self.stats[provider.name]["fail"] += 1
                    self.stats[provider.name]["errors"].append(str(error))
                    self._log(f"  [{provider.name}] rate limited — disabling for the rest of this run.")
                except AuthError as error:
                    self._disabled.add(provider.name)
                    self.stats[provider.name]["fail"] += 1
                    self.stats[provider.name]["errors"].append(str(error))
                    self._log(f"  [{provider.name}] {error} — disabling.")
                except HttpError as error:
                    self.stats[provider.name]["fail"] += 1
                    self.stats[provider.name]["errors"].append(str(error))
                    self._log(f"  [{provider.name}] {error}")
                except Exception as error:  # never let one provider abort the run
                    self.stats[provider.name]["fail"] += 1
                    self.stats[provider.name]["errors"].append(repr(error))
                    self._log(f"  [{provider.name}] unexpected error: {error!r}")

        merged = _merge(per_provider)
        for result in merged:
            result.signal_category = signal_category
        # Cache only a COMPLETE result for this key's provider set — i.e. every
        # active provider succeeded. If one failed mid-call, skip caching so a
        # partial set is never served later as if it were complete.
        complete = set(succeeded) == set(active_names)
        if self.cache is not None and merged and complete:
            self.cache.put_search(cache_key, merged)
        return merged

    def all_disabled(self) -> bool:
        return bool(self.providers) and len(self._disabled) == len(self.providers)
