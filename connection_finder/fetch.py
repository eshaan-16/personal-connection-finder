from __future__ import annotations

import html
import re
import time
from html.parser import HTMLParser

from .dates import extract_date_from_html
from .http_util import HttpError, request_bytes
from .search.base import is_public_http_url

# Skip site chrome and sidebars. <aside> in particular holds "related articles",
# "you might also like", and promo widgets — the usual source of off-topic names.
_SKIP_TAGS = {"script", "style", "svg", "noscript", "head", "nav", "footer",
              "form", "aside"}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if self.skip_depth:
            return
        cleaned = " ".join(data.split())
        if cleaned:
            self.parts.append(cleaned)

    def text(self) -> str:
        return " ".join(self.parts)


def fetch_page(
    url: str,
    *,
    max_chars: int = 14000,
    timeout: float = 20.0,
    delay: float = 0.2,
    allow_insecure_ssl: bool = False,
    collect_images: bool = False,
) -> tuple[str, str, list]:
    """Return (clean_text, iso_published_date, images) for a public URL.

    ``images`` is a list of ImageRef (empty unless ``collect_images``). Failures
    (timeouts, blocks, non-HTML) return ("", "", []) rather than raising; the
    snippet from the search result still carries the citation.
    """
    if not is_public_http_url(url):
        return "", "", []
    try:
        raw, headers = request_bytes(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; ArtemisConnectionFinder/0.1; +public-source-research)",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=timeout,
            retries=2,
            allow_insecure_ssl=allow_insecure_ssl,
            label="fetch",
            max_bytes=max_chars * 6,
        )
    except HttpError:
        return "", "", []
    except Exception:
        return "", "", []

    content_type = (headers.get("Content-Type") or headers.get("content-type") or "").lower()
    if content_type and "html" not in content_type and "text/plain" not in content_type:
        return "", "", []

    charset = "utf-8"
    match = re.search(r"charset=([\w-]+)", content_type)
    if match:
        charset = match.group(1)
    html_text = raw.decode(charset, errors="replace")

    published = extract_date_from_html(html_text, url)

    parser = _TextExtractor()
    try:
        parser.feed(html_text)
    except Exception:
        pass
    text = html.unescape(parser.text())
    text = re.sub(r"\s+", " ", text).strip()[:max_chars]

    images: list = []
    if collect_images:
        from .images import extract_images
        images = extract_images(html_text, url)
        # Names sometimes live only in an image caption/alt, which the body-text
        # extractor never sees. Fold captioned images' text back in.
        captions = [ref.caption_text() for ref in images if ref.has_caption()]
        if captions:
            text = (text + " " + " ".join(captions[:15])).strip()

    if delay:
        time.sleep(delay)
    return text, published, images
