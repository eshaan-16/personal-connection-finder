#!/usr/bin/env python3
"""Launch the local web UI:

    python serve.py            # http://127.0.0.1:8000
    python serve.py --port 8080
"""
from __future__ import annotations

import argparse

from connection_finder.webui import serve


def main() -> int:
    parser = argparse.ArgumentParser(description="Connection Finder web UI (local).")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default 8000)")
    args = parser.parse_args()
    return serve(host=args.host, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())
