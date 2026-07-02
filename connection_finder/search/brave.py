from __future__ import annotations

from urllib.parse import quote_plus

from ..dates import to_iso_date
from ..http_util import request_json
from ..models import SearchResult
from ..util import squeeze
from .base import SearchProvider, is_public_http_url


class BraveProvider(SearchProvider):
    name = "brave"

    def __init__(self, api_key: str, *, retries: int = 4, allow_insecure_ssl: bool = False):
        self.api_key = api_key
        self.retries = retries
        self.allow_insecure_ssl = allow_insecure_ssl

    def search(self, query: str, count: int) -> list[SearchResult]:
        url = (
            "https://api.search.brave.com/res/v1/web/search"
            f"?q={quote_plus(query)}&count={min(max(count, 1), 20)}"
            "&text_decorations=false&result_filter=web"
        )
        data = request_json(
            url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "X-Subscription-Token": self.api_key,
            },
            retries=self.retries,
            allow_insecure_ssl=self.allow_insecure_ssl,
            label="brave",
        )
        results: list[SearchResult] = []
        for rank, item in enumerate(data.get("web", {}).get("results", []), 1):
            result_url = item.get("url", "")
            title = squeeze(item.get("title", ""))
            if not (title and is_public_http_url(result_url)):
                continue
            published = to_iso_date(item.get("page_age", "") or item.get("age", ""))
            results.append(self._make_result(
                query=query, provider=self.name, title=title, url=result_url,
                snippet=squeeze(item.get("description", "")), rank=rank,
                published_date=published,
            ))
        return results
