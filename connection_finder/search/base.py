from __future__ import annotations

import ipaddress
import socket
from abc import ABC, abstractmethod
from urllib.parse import urlparse

from ..models import SearchResult

# Hostnames/strings that are always unwanted regardless of IP resolution.
_BLOCKED_HOST_HINTS = (
    "accounts.google.com", "support.google.com", "policies.google.com",
    "login.", "signin.",
)


def _addr_is_blocked(addr) -> bool:
    return (addr.is_private or addr.is_loopback or addr.is_link_local
            or addr.is_reserved or addr.is_multicast or addr.is_unspecified)


def ip_is_blocked(ip_str: str) -> bool:
    """Return True if a raw IP string is a private/loopback/link-local/reserved/
    multicast/unspecified address. Used for the authoritative connect-time check
    (see http_util._guarded_create_connection) so a validated hostname cannot be
    rebound to a private IP between the check and the actual socket connect."""
    try:
        return _addr_is_blocked(ipaddress.ip_address(ip_str.strip("[]")))
    except ValueError:
        return False


def _ip_is_private(host: str) -> bool:
    """Return True if ``host`` (already stripped of port) resolves to / is a
    private, loopback, link-local, reserved, or multicast address.

    Catches: 127.x.x.x, localhost, ::1, 169.254.x.x (metadata), 10/8,
    172.16/12, 192.168/16, fc00::/7, etc. This is a cheap first gate; the
    authoritative defence against DNS rebinding is the connect-time IP pin.
    """
    # Normalize bare bracketed IPv6 — urlparse leaves the brackets.
    ip_candidate = host.strip("[]")
    # Try direct parse first (handles numeric IPv4, decimal, dotted, IPv6).
    try:
        return _addr_is_blocked(ipaddress.ip_address(ip_candidate))
    except ValueError:
        pass
    # Hostname — resolve and check every returned address.
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        for _, _, _, _, sockaddr in infos:
            if ip_is_blocked(sockaddr[0]):
                return True
    except (socket.gaierror, OSError):
        # DNS failure — treat as unparseable, not as private; let the fetch
        # itself fail naturally (it will) rather than blocking on DNS error.
        pass
    return False


def is_public_http_url(url: str) -> bool:
    """Return True only for public HTTP(S) URLs safe to fetch.

    Blocks: non-http(s) schemes, missing host, known policy/login subdomains,
    loopback, link-local (169.254/16 — cloud metadata), RFC1918 private ranges,
    IPv6 ULA/loopback, multicast, numeric-encoded private IPs.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    # parsed.netloc includes port; parsed.hostname strips port+brackets.
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if any(hint in host for hint in _BLOCKED_HOST_HINTS):
        return False
    if _ip_is_private(host):
        return False
    return True


class SearchProvider(ABC):
    """One search backend. ``search`` may raise an http_util.HttpError subclass;
    the engine catches those per-provider so one failure never aborts a run."""

    name: str = "provider"

    @abstractmethod
    def search(self, query: str, count: int) -> list[SearchResult]:
        ...

    def _make_result(self, *, query, provider, title, url, snippet, rank, published_date="") -> SearchResult:
        return SearchResult(
            query=query,
            signal_category="",  # filled in by the caller from the query plan
            provider=provider,
            title=title,
            url=url,
            snippet=snippet,
            rank=rank,
            published_date=published_date,
        )
