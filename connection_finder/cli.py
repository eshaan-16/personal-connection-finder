from __future__ import annotations

import argparse
import re
import sys

from .config import ConfigError, Settings
from .pipeline import RunResult, find_connectors
from .report import render_console, write_json, write_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="connection-finder",
        description="Find public-source warm-introduction paths (family, friends, "
                    "close associates) to a target person.",
    )
    parser.add_argument("target", nargs="?", help="Target person's name, e.g. \"Bill Gates\"")
    parser.add_argument("--context", "-c", help="Disambiguation string, e.g. \"Microsoft\" (required)")
    parser.add_argument("--location", help="Optional: narrow toward a location")
    parser.add_argument("--industry", help="Optional: narrow toward an industry")
    parser.add_argument("--period", help="Optional: narrow toward an era, e.g. \"2010s\"")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode: choose the person and context for each search, "
                             "and keep searching more people in one session")

    parser.add_argument("--connections", help="Path to your LinkedIn Connections.csv (network index)")
    parser.add_argument("--second-degree", help="Optional JSON map {connection_name: [their_connections]}")

    parser.add_argument("--db", default="connection_finder.sqlite3", help="Standing SQLite DB path")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the persistent search cache")

    parser.add_argument("--providers", help="Comma list to restrict providers: brave,google_cse,bing")
    parser.add_argument("--max-results", type=int, default=6, help="Results per query (default 6)")
    parser.add_argument("--max-pages", type=int, default=3, help="Pages to fetch full text per query (default 3)")
    parser.add_argument("--max-queries", type=int, default=0, help="Cap total queries (0 = full batch)")
    parser.add_argument("--no-fetch", action="store_true", help="Skip page fetch; use snippets only (faster/cheaper)")
    parser.add_argument("--analyze-photos", action="store_true",
                        help="Vision-analyze uncaptioned photos for named people (needs GEMINI_API_KEY)")
    parser.add_argument("--max-photos", type=int, default=4, help="Max photos to vision-analyze per run (default 4)")
    parser.add_argument("--stale-years", type=int, default=15, help="Penalize sole evidence older than this")

    parser.add_argument("--out", help="Write a Markdown report to this path")
    parser.add_argument("--json", dest="json_out", help="Write JSON output to this path")
    parser.add_argument("--limit", type=int, default=25, help="Console rows to print (default 25)")
    parser.add_argument("--allow-insecure-ssl", action="store_true", help="Disable TLS verification (last resort)")
    return parser


def _build_settings(args) -> Settings:
    only = [p.strip() for p in args.providers.split(",")] if args.providers else []
    return Settings.from_env(
        max_results_per_query=args.max_results,
        max_pages_per_query=args.max_pages,
        max_queries=args.max_queries,
        fetch_pages=not args.no_fetch,
        analyze_photos=args.analyze_photos,
        max_photos=args.max_photos,
        stale_years=args.stale_years,
        db_path=args.db,
        use_cache=not args.no_cache,
        connections_csv=args.connections or "",
        second_degree_json=args.second_degree or "",
        allow_insecure_ssl=args.allow_insecure_ssl,
        only_providers=only,
    )


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "target"


def _emit(result: RunResult, args, *, suffix: str = "") -> None:
    print(render_console(result, limit=args.limit))
    if args.out:
        path = args.out
        if suffix and "." in path:
            base, ext = path.rsplit(".", 1)
            path = f"{base}-{suffix}.{ext}"
        elif suffix:
            path = f"{path}-{suffix}"
        write_markdown(result, path)
        print(f"\nMarkdown report: {path}", file=sys.stderr)
    if args.json_out:
        path = args.json_out
        if suffix and "." in path:
            base, ext = path.rsplit(".", 1)
            path = f"{base}-{suffix}.{ext}"
        elif suffix:
            path = f"{path}-{suffix}"
        write_json(result, path)
        print(f"JSON output: {path}", file=sys.stderr)


def _run_one(settings: Settings, args, *, target: str, context: str,
             location=None, industry=None, period=None, suffix: str = "") -> int:
    try:
        result = find_connectors(
            settings, target, context,
            location=location, industry=industry, period=period,
        )
    except ConfigError as error:
        print(f"\n{error}", file=sys.stderr)
        return 3
    _emit(result, args, suffix=suffix)
    return 0


def _interactive_loop(settings: Settings, args) -> int:
    print("Connection Finder — interactive mode.")
    print("For each person, enter their name and a disambiguation context "
          "(company, school, city — anything that pins down which person).")
    print("Leave the name blank to quit.\n")
    ran_any = False
    while True:
        target = _ask("Who do you want to search? (name, blank to quit): ")
        if not target:
            break
        # Context is required to pin down the right person; re-ask once, then
        # let the user skip this person by leaving it blank a second time.
        context = _ask("Context to disambiguate (e.g. Microsoft, Stanford CS): ")
        if not context:
            context = _ask("Context is required — enter one, or leave blank to skip this person: ")
        if not context:
            print("Skipped.\n")
            continue
        location = _ask("Optional location (blank to skip): ") or None
        industry = _ask("Optional industry (blank to skip): ") or None
        period = _ask("Optional era, e.g. 2010s (blank to skip): ") or None
        print()
        _run_one(settings, args, target=target, context=context,
                 location=location, industry=industry, period=period, suffix=_slug(target))
        ran_any = True
        print("\n" + "=" * 78 + "\n")
    if not ran_any:
        print("No searches run.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = _build_settings(args)

    # Interactive when asked for, or when no target was supplied on the command line.
    if args.interactive or not args.target:
        return _interactive_loop(settings, args)

    context = args.context
    if not context:
        context = _ask("Disambiguation context (e.g. Microsoft, Stanford CS): ")
    if not context:
        print("A context string is required to disambiguate the target.", file=sys.stderr)
        return 2

    return _run_one(
        settings, args, target=args.target.strip(), context=context,
        location=args.location, industry=args.industry, period=args.period,
    )


if __name__ == "__main__":
    raise SystemExit(main())
