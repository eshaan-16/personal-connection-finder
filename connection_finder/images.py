from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urljoin

from .http_util import HttpError, request_bytes
from .search.base import is_public_http_url
from .util import squeeze

_GENERIC_CAPTION = {
    "image", "images", "photo", "photos", "picture", "pictures", "logo",
    "icon", "avatar", "thumbnail", "banner", "graphic", "photograph", "img",
    "figure", "loading", "placeholder", "spacer", "advertisement", "ad",
}

_NON_PHOTO_HINTS = (
    "logo", "icon", "sprite", "favicon", "avatar", "placeholder", "spacer",
    "1x1", "pixel", "tracking", "badge", "button", "banner-ad", "/ads/",
    ".svg", ".gif",
)

_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif")
_MIME_BY_EXT = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".heic": "image/heic", ".heif": "image/heif",
}


@dataclass
class ImageRef:
    url: str
    alt: str = ""
    caption: str = ""
    title: str = ""
    page_url: str = ""

    def caption_text(self) -> str:
        return squeeze(" ".join(filter(None, [self.alt, self.caption, self.title])))

    def has_caption(self) -> bool:
        text = self.caption_text()
        words = re.findall(r"[a-z]+", text.lower())
        if len(words) < 2:
            return False
        return not all(word in _GENERIC_CAPTION for word in words)


def _looks_like_photo(url: str) -> bool:
    low = url.lower()
    if low.startswith("data:"):
        return False
    if any(hint in low for hint in _NON_PHOTO_HINTS):
        return False
    path = low.split("?", 1)[0]
    if any(path.endswith(ext) for ext in _IMAGE_EXT):
        return True
    return "." not in path.rsplit("/", 1)[-1]


class _ImageCollector(HTMLParser):
    """HTML parser that collects image references with caption association.

    Uses a stack of per-level image lists to correctly associate captions with
    the images in their immediate <figure>, even when figures are nested.
    """

    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.images: list[ImageRef] = []
        # Stack of per-nesting-level image lists for nested <figure> elements.
        self._figure_stack: list[list[ImageRef]] = []
        self._in_figcaption = False
        self._figcaption_parts: list[str] = []
        self._og_image = ""

    def handle_starttag(self, tag, attrs):
        attr = {k.lower(): (v or "") for k, v in attrs}
        if tag == "meta":
            prop = attr.get("property", "") or attr.get("name", "")
            if prop.lower() in ("og:image", "twitter:image", "og:image:url") and attr.get("content"):
                self._og_image = attr["content"]
            return
        if tag == "figure":
            self._figure_stack.append([])  # push a new level
            return
        if tag == "figcaption":
            self._in_figcaption = True
            self._figcaption_parts = []
            return
        if tag == "img":
            src = attr.get("src") or attr.get("data-src") or attr.get("data-original") or ""
            if not src:
                return
            ref = ImageRef(
                url=urljoin(self.base_url, src),
                alt=squeeze(attr.get("alt", "")),
                title=squeeze(attr.get("title", "")),
                page_url=self.base_url,
            )
            if self._figure_stack:
                self._figure_stack[-1].append(ref)  # add to innermost figure
            else:
                self.images.append(ref)

    def handle_endtag(self, tag):
        if tag == "figcaption":
            self._in_figcaption = False
            caption = squeeze(" ".join(self._figcaption_parts))
            # Apply caption only to images at the CURRENT (innermost) figure level.
            if self._figure_stack:
                for ref in self._figure_stack[-1]:
                    if not ref.caption:
                        ref.caption = caption
            return
        if tag == "figure" and self._figure_stack:
            level_imgs = self._figure_stack.pop()
            if self._figure_stack:
                # Nested figure: bubble images up to the parent level.
                self._figure_stack[-1].extend(level_imgs)
            else:
                # Outermost figure closed: commit all its images to the main list.
                self.images.extend(level_imgs)

    def handle_data(self, data):
        if self._in_figcaption:
            self._figcaption_parts.append(data)

    def finalize(self) -> list[ImageRef]:
        # Flush images from any unclosed figure elements.
        for level_imgs in self._figure_stack:
            self.images.extend(level_imgs)
        self._figure_stack = []
        if self._og_image:
            self.images.insert(0, ImageRef(
                url=urljoin(self.base_url, self._og_image), page_url=self.base_url))
        return self.images


def extract_images(html_text: str, base_url: str, *, max_images: int = 25) -> list[ImageRef]:
    if not html_text:
        return []
    collector = _ImageCollector(base_url)
    try:
        collector.feed(html_text)
    except Exception:
        pass
    seen: set[str] = set()
    output: list[ImageRef] = []
    for ref in collector.finalize():
        if not ref.url or not is_public_http_url(ref.url) or not _looks_like_photo(ref.url):
            continue
        key = ref.url.split("#", 1)[0]
        if key in seen:
            continue
        seen.add(key)
        output.append(ref)
        if len(output) >= max_images:
            break
    return output


def _magic_mime(data: bytes) -> Optional[str]:
    """Detect MIME type from leading magic bytes."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"\x89PNG":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _mime_for(url: str, content_type: str, data: bytes = b"") -> Optional[str]:
    """Determine MIME type for a downloaded image.

    Trust order: Content-Type (if a known image type) → magic bytes (when data
    available) → file extension. Extension is only used as last resort because
    a server can lie with a non-matching extension; magic bytes are authoritative.
    """
    ctype = (content_type or "").split(";", 1)[0].strip().lower()
    if ctype in ("image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"):
        return ctype
    # Content-Type is absent or not a recognized image type — check magic bytes.
    if data:
        magic = _magic_mime(data)
        if magic:
            return magic
    # Fall back to file extension.
    path = url.lower().split("?", 1)[0]
    for ext, mime in _MIME_BY_EXT.items():
        if path.endswith(ext):
            return mime
    # Last resort: accept any image/* Content-Type we didn't recognise above.
    if ctype.startswith("image/"):
        return ctype
    return None


def download_image(url: str, *, max_bytes: int = 5_000_000, timeout: float = 20.0,
                   allow_insecure_ssl: bool = False) -> Optional[tuple[bytes, str]]:
    """Fetch image bytes + mime type, or None on any failure / non-image / oversize."""
    if not is_public_http_url(url):
        return None
    try:
        raw, headers = request_bytes(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ArtemisConnectionFinder/0.1)"},
            timeout=timeout, retries=2, allow_insecure_ssl=allow_insecure_ssl,
            label="image", max_bytes=max_bytes + 1,
        )
    except HttpError:
        return None
    except Exception:
        return None
    if not raw or len(raw) > max_bytes:
        return None
    ct = headers.get("Content-Type") or headers.get("content-type") or ""
    mime = _mime_for(url, ct, raw)
    if not mime:
        return None
    return raw, mime
