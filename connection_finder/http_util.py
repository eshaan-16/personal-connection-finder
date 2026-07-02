from __future__ import annotations

import email.utils
import http.client
import json
import socket
import ssl
import time
import urllib.request
from datetime import datetime, timezone
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request


class HttpError(RuntimeError):
    """Base class for outbound-request failures with a human-readable message."""


class RateLimitError(HttpError):
    """429 / quota exhausted after retries. Carries the provider for messaging."""


class AuthError(HttpError):
    """401 / 403 — almost always a missing or invalid API key."""


class TransientError(HttpError):
    """Network / 5xx failure that persisted across retries."""


def ssl_context(allow_insecure: bool = False):
    if allow_insecure:
        return ssl._create_unverified_context()
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def _guarded_create_connection(host, port, timeout, source_address):
    """Resolve ``host`` and connect ONLY to a validated public address.

    Unlike socket.create_connection (which the SSRF check cannot see into), this
    validates every resolved address and connects to the exact validated
    sockaddr — closing the DNS-rebinding TOCTOU window where a hostname passes
    is_public_http_url and then re-resolves to 169.254.169.254 / 127.0.0.1 at
    connect time. Also blocks numeric-encoded IPs the platform resolver
    normalizes (e.g. 2130706433 -> 127.0.0.1).
    """
    from .search.base import ip_is_blocked  # lazy: avoid search-package import cycle

    infos = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    last_error = None
    for family, socktype, proto, _canon, sockaddr in infos:
        if ip_is_blocked(sockaddr[0]):
            raise HttpError(f"blocked non-public address {sockaddr[0]} for host {host!r}")
    for family, socktype, proto, _canon, sockaddr in infos:
        sock = None
        try:
            sock = socket.socket(family, socktype, proto)
            if isinstance(timeout, (int, float)):
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)  # pin to the exact address we validated
            return sock
        except OSError as error:
            last_error = error
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
    if last_error is not None:
        raise last_error
    raise HttpError(f"could not connect to {host!r}")


class _GuardedHTTPConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = _guarded_create_connection(
            self.host, self.port, self.timeout, self.source_address)


class _GuardedHTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        sock = _guarded_create_connection(
            self.host, self.port, self.timeout, self.source_address)
        # Keep the original hostname for SNI / certificate validation.
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class _GuardedHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_GuardedHTTPConnection, req)


class _GuardedHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_GuardedHTTPSConnection, req, context=self._context)


class _SSRFRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validates each redirect target before following it.

    Prevents open-redirect exploitation where a public URL redirects to a
    private IP (169.254.x.x metadata services, 10.x, 127.x.x.x, etc.).
    The import is lazy so the module loads without a circular-import issue.
    """
    max_repeats = 4
    max_redirections = 4

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        from .search.base import is_public_http_url  # lazy to avoid circular import
        if not is_public_http_url(newurl):
            raise HttpError(f"Redirect to private/blocked host rejected: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _retry_after_seconds(error: HTTPError, attempt: int, base: float) -> float:
    header = error.headers.get("Retry-After") if error.headers else None
    if header:
        try:
            return min(float(header), 30.0)
        except (TypeError, ValueError):
            pass
        try:
            then = email.utils.parsedate_to_datetime(header)
            if then is not None:
                now = datetime.now(then.tzinfo or timezone.utc)
                return max(0.0, min((then - now).total_seconds(), 30.0))
        except (TypeError, ValueError, OverflowError):
            pass
    return min(base * (2 ** attempt), 30.0)


def _build_opener(context) -> urllib.request.OpenerDirector:
    """Build an opener with SSRF-safe redirect handling, connect-time IP pinning,
    and the optional SSL context. The guarded HTTP(S) handlers replace urllib's
    defaults (they subclass them), so every connection validates its peer IP."""
    handlers: list = [_SSRFRedirectHandler(), _GuardedHTTPHandler()]
    if context is not None:
        handlers.append(_GuardedHTTPSHandler(context=context))
    else:
        handlers.append(_GuardedHTTPSHandler())
    return urllib.request.build_opener(*handlers)


def request_bytes(
    url: str,
    *,
    headers: Optional[dict] = None,
    data: Optional[bytes] = None,
    method: Optional[str] = None,
    timeout: float = 30.0,
    retries: int = 4,
    backoff_base: float = 1.0,
    allow_insecure_ssl: bool = False,
    label: str = "request",
    max_bytes: Optional[int] = None,
) -> tuple[bytes, dict]:
    """Perform an HTTP request with retry/backoff on 429 and 5xx.

    Returns (body_bytes, response_headers). Raises a typed HttpError subclass on
    persistent failure. Redirect targets are validated to prevent SSRF.
    """
    context = ssl_context(allow_insecure_ssl)
    opener = _build_opener(context)
    last_error: Optional[Exception] = None

    for attempt in range(max(1, retries)):
        request = Request(url, data=data, headers=headers or {}, method=method)
        try:
            with opener.open(request, timeout=timeout) as response:
                raw = response.read(max_bytes) if max_bytes else response.read()
                return raw, dict(response.headers.items())
        except HTTPError as error:
            last_error = error
            status = error.code
            if status in (401, 403):
                raise AuthError(
                    f"{label}: authentication failed (HTTP {status}). "
                    f"Check the API key."
                ) from error
            if status == 429:
                wait = _retry_after_seconds(error, attempt, backoff_base)
                if attempt + 1 < retries:
                    time.sleep(wait)
                    continue
                raise RateLimitError(
                    f"{label}: rate limit / quota exceeded (HTTP 429) after "
                    f"{retries} attempts. Back off and retry later."
                ) from error
            if 500 <= status < 600:
                if attempt + 1 < retries:
                    time.sleep(min(backoff_base * (2 ** attempt), 20.0))
                    continue
                raise TransientError(
                    f"{label}: server error (HTTP {status}) after {retries} attempts."
                ) from error
            raise HttpError(f"{label}: HTTP {status} {error.reason}") from error
        except (URLError, TimeoutError, OSError, http.client.HTTPException) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(backoff_base * (2 ** attempt), 20.0))
                continue
            raise TransientError(f"{label}: network error: {error}") from error

    raise TransientError(f"{label}: failed: {last_error}")


def request_json(url: str, **kwargs) -> dict:
    raw, _ = request_bytes(url, **kwargs)
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as error:
        raise HttpError(f"{kwargs.get('label', 'request')}: invalid JSON response") from error
