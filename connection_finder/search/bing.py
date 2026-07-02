from __future__ import annotations

from urllib.parse import quote_plus

from ..dates import to_iso_date
from ..http_util import request_json
from ..models import SearchResult
from ..util import squeeze
from .base import SearchProvider, is_public_http_url


class BingProvider(SearchProvider):
    """Bing Web Search v7 (Azure).

    Note: Microsoft has announced retirement of the Bing Search APIs. This
    provider is kept for users who still hold a working key; the engine treats
    its failure like any other provider and degrades gracefully.
    """
    name = "bing"

    def __init__(self, api_key: str, *, endpoint: str = "https://api.bing.microsoft.com/v7.0/search",
                 retries: int = 4, allow_insecure_ssl: bool = False):
        self.api_key = api_key
        self.endpoint = endpoint
        self.retries = retries
        self.allow_insecure_ssl = allow_insecure_ssl

    def search(self, query: str, count: int) -> list[SearchResult]:
        url = (
            f"{self.endpoint}?q={quote_plus(query)}"
            f"&count={min(max(count, 1), 20)}&responseFilter=Webpages&textDecorations=false"
        )
        data = request_json(
            url,
            headers={"Ocp-Apim-Subscription-Key": self.api_key},
            retries=self.retries,
            allow_insecure_ssl=self.allow_insecure_ssl,
            label="bing",
        )
        results: list[SearchResult] = []
        values = (data.get("webPages") or {}).get("value", []) or []
        for rank, item in enumerate(values, 1):
            result_url = item.get("url", "")
            title = squeeze(item.get("name", ""))
            if not (title and is_public_http_url(result_url)):
                continue
            # Only the genuine publication date. dateLastCrawled is the crawl
            # timestamp (≈ now) and would make stale evidence look fresh; leave
            # the date empty and let page-level extraction supply a real one.
            published = to_iso_date(item.get("datePublished", ""))
            results.append(self._make_result(
                query=query, provider=self.name, title=title, url=result_url,
                snippet=squeeze(item.get("snippet", "")), rank=rank,
                published_date=published,
            ))
        return results
