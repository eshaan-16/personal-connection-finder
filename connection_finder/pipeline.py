from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional

from .config import Settings
from .extract import GeminiExtractor, RawCandidate, extract_candidates, photo_candidates
from .fetch import fetch_page
from .images import ImageRef
from .models import ScoredCandidate, SearchResult
from .network import NetworkIndex, build_index
from .pricing import CostEstimate, estimate_cost
from .query import build_queries
from .score import score_and_rank
from .search import SearchEngine, build_providers
from .store import Store
from .util import normalize_name


@dataclass
class RunResult:
    target: str
    context: str
    scored: list[ScoredCandidate]
    queries_run: int
    results_seen: int
    providers: list[str]
    extractor: str
    provider_stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    cost: Optional[CostEstimate] = None


def _log(message: str, verbose: bool) -> None:
    if verbose:
        print(message, file=sys.stderr)


def find_connectors(
    settings: Settings,
    target: str,
    context: str,
    *,
    location: Optional[str] = None,
    industry: Optional[str] = None,
    period: Optional[str] = None,
    verbose: bool = True,
) -> RunResult:
    """Run the full discovery pipeline and return ranked candidate connectors.

    Steps: build queries -> multi-provider search -> fetch/parse pages ->
    extract people -> dedupe + score (recency/corroboration/network) -> persist.
    """
    settings.validate_for_search()
    target = target.strip()
    context = context.strip()
    target_key = normalize_name(target)
    warnings: list[str] = []

    store = Store(settings.db_path, cache_ttl_hours=settings.cache_ttl_hours, use_cache=settings.use_cache)
    try:
        providers = build_providers(settings)
        engine = SearchEngine(providers, cache=store, max_workers=max(1, len(providers)), verbose=verbose)
        gemini = (
            GeminiExtractor(settings.gemini_api_key, settings.gemini_model,
                            retries=settings.http_retries, allow_insecure_ssl=settings.allow_insecure_ssl)
            if settings.has_gemini() else None
        )
        extractor_name = "gemini" if gemini else "heuristic"

        network = build_index(settings.connections_csv, settings.second_degree_json, verbose=verbose)
        if settings.connections_csv and network.size[0] == 0:
            warnings.append("Network index is empty; in-network flags will be unavailable.")

        queries = build_queries(
            target, context, location=location, industry=industry, period=period,
            limit=settings.max_queries,
        )
        _log(f"Providers: {', '.join(engine.provider_names) or 'none'} | extractor: {extractor_name}", verbose)
        _log(f"Running {len(queries)} queries across {len(providers)} provider(s)...", verbose)

        results_by_url: dict[str, SearchResult] = {}
        raw_candidates: list[RawCandidate] = []
        results_seen = 0
        want_photos = settings.analyze_photos and gemini is not None
        photo_pool: list[tuple[ImageRef, str]] = []
        photo_seen: set[str] = set()
        if settings.analyze_photos and gemini is None:
            warnings.append("Photo analysis needs a GEMINI_API_KEY; skipping image analysis.")

        run_id = store.start_run(
            target=target, target_key=target_key, context=context, location=location,
            industry=industry, period=period, providers=engine.provider_names, extractor=extractor_name,
        )

        for spec in queries:
            if engine.all_disabled():
                warnings.append("All search providers were rate-limited or failed; stopping early.")
                _log("All providers disabled — stopping query loop.", verbose)
                break
            _log(f"  search [{spec.signal_category}] {spec.text}", verbose)
            try:
                results = engine.search(spec.text, settings.max_results_per_query, spec.signal_category)
            except Exception as error:  # engine already handles per-provider errors
                warnings.append(f"Search failed for {spec.text!r}: {error}")
                continue
            results_seen += len(results)

            batch: list[SearchResult] = []
            for index, result in enumerate(results):
                # Fetch full text for the top N results of each query (cheaper than
                # fetching everything; snippets still cite the rest).
                if settings.fetch_pages and index < settings.max_pages_per_query and not result.page_text:
                    text, published, images = fetch_page(
                        result.url,
                        delay=settings.request_delay,
                        allow_insecure_ssl=settings.allow_insecure_ssl,
                        collect_images=want_photos,
                    )
                    result.page_text = text
                    if not result.published_date and published:
                        result.published_date = published
                    for image in images:
                        key = image.url.split("#", 1)[0]
                        if key not in photo_seen:
                            photo_seen.add(key)
                            photo_pool.append((image, result.published_date))
                results_by_url[result.url] = result
                batch.append(result)

            if not batch:
                continue
            extracted = extract_candidates(target, context, batch, gemini=gemini, verbose=verbose)
            raw_candidates.extend(extracted)

        # One bounded vision pass over uncaptioned photos gathered this run.
        if want_photos and photo_pool:
            uncaptioned = [(img, date) for img, date in photo_pool if not img.has_caption()]
            _log(f"Analyzing up to {settings.max_photos} of {len(uncaptioned)} uncaptioned photo(s)...", verbose)
            budget = settings.max_photos
            for image, page_date in uncaptioned:
                if budget <= 0:
                    break
                budget -= 1
                raw_candidates.extend(
                    photo_candidates(target, context, [image], gemini=gemini,
                                     max_photos=1, page_date=page_date, verbose=verbose)
                )

        scored = score_and_rank(
            raw_candidates, results_by_url, network=network,
            stale_years=settings.stale_years, recent_years=settings.recent_years,
        )
        store.record(run_id, target_key, scored)
    finally:
        store.close()  # always release the sqlite/WAL handle, even on error

    if not engine.provider_names:
        warnings.append("No providers ran.")

    # Estimate what this run cost: live search calls (cache hits are free) plus
    # actual Gemini token usage reported by the API.
    search_calls = {name: stat.get("ok", 0) for name, stat in engine.stats.items() if stat.get("ok", 0)}
    cost = estimate_cost(
        model=settings.gemini_model if gemini else "",
        gemini_calls=gemini.calls if gemini else 0,
        prompt_tokens=gemini.prompt_tokens if gemini else 0,
        output_tokens=gemini.output_tokens if gemini else 0,
        search_calls_by_provider=search_calls,
    )

    return RunResult(
        target=target,
        context=context,
        scored=scored,
        queries_run=len(queries),
        results_seen=results_seen,
        providers=engine.provider_names,
        extractor=extractor_name,
        provider_stats=engine.stats,
        warnings=warnings,
        cost=cost,
    )
