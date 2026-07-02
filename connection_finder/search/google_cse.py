from __future__ import annotations

from urllib.parse import quote_plus

from ..dates import to_iso_date
from ..http_util import request_json
from ..models import SearchResult
from ..util import squeeze
from .base import SearchProvider, is_public_http_url


def _date_from_pagemap(item: dict) -> str:
    pagemap = item.get("pagemap") or {}
    metatags = (pagemap.get("metatags") or [{}])[0]
    for key in (
        "article:published_time", "article:modified_time", "og:updated_time",
        "datepublished", "date", "dc.date", "sailthru.date",
    ):
        value = metatags.get(key)
        if value:
            iso = to_iso_date(value)
            if iso:
                return iso
    for block_key in ("newsarticle", "article", "webpage"):
        for block in pagemap.get(block_key, []) or []:
            for date_key in ("datepublished", "datemodified", "datecreated"):
                if block.get(date_key):
                    iso = to_iso_date(block[date_key])
                    if iso:
                        return iso
    return ""


class GoogleCseProvider(SearchProvider):
    """Google Programmable Search Engine (Custom Search JSON API)."""
    name = "google_cse"

    def __init__(self, cse_id: str, api_key: str, *, retries: int = 4, allow_insecure_ssl: bool = False):
        self.cse_id = cse_id
        self.api_key = api_key
        self.retries = retries
        self.allow_insecure_ssl = allow_insecure_ssl

    def search(self, query: str, count: int) -> list[SearchResult]:
        # CSE caps num at 10 per call.
        url = (
            "https://www.googleapis.com/customsearch/v1"
            f"?key={quote_plus(self.api_key)}&cx={quote_plus(self.cse_id)}"
            f"&q={quote_plus(query)}&num={min(max(count, 1), 10)}"
        )
        data = request_json(
            url, retries=self.retries,
            allow_insecure_ssl=self.allow_insecure_ssl, label="google_cse",
        )
        results: list[SearchResult] = []
        for rank, item in enumerate(data.get("items", []) or [], 1):
            result_url = item.get("link", "")
            title = squeeze(item.get("title", ""))
            if not (title and is_public_http_url(result_url)):
                continue
            results.append(self._make_result(
                query=query, provider=self.name, title=title, url=result_url,
                snippet=squeeze(item.get("snippet", "")), rank=rank,
                published_date=_date_from_pagemap(item),
            ))
        return results
