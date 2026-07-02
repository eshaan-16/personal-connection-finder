from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .models import ScoredCandidate, SearchResult, best_signal

_SCHEMA = """
CREATE TABLE IF NOT EXISTS search_cache (
    cache_key   TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    payload     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    target      TEXT NOT NULL,
    target_key  TEXT NOT NULL,
    context     TEXT,
    location    TEXT,
    industry    TEXT,
    period      TEXT,
    providers   TEXT,
    extractor   TEXT,
    n_candidates INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidates (
    name_key    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS connections (
    candidate_key   TEXT NOT NULL,
    target_key      TEXT NOT NULL,
    signal_category TEXT,
    explanation     TEXT,
    best_confidence REAL DEFAULT 0,
    first_seen      TEXT NOT NULL,
    last_seen       TEXT NOT NULL,
    PRIMARY KEY (candidate_key, target_key)
);
CREATE TABLE IF NOT EXISTS sources (
    candidate_key   TEXT NOT NULL,
    target_key      TEXT NOT NULL,
    url             TEXT NOT NULL,
    domain          TEXT,
    title           TEXT,
    snippet         TEXT,
    published_date  TEXT,
    provider        TEXT,
    query           TEXT,
    signal_category TEXT,
    run_id          INTEGER,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (candidate_key, target_key, url)
);
CREATE TABLE IF NOT EXISTS scores (
    run_id          INTEGER NOT NULL,
    candidate_key   TEXT NOT NULL,
    target_key      TEXT NOT NULL,
    score           REAL,
    tier            TEXT,
    in_network      INTEGER,
    degree          INTEGER,
    via             TEXT,
    distinct_domains INTEGER,
    most_recent_date TEXT,
    stale_only      INTEGER,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (run_id, candidate_key, target_key)
);
CREATE INDEX IF NOT EXISTS idx_sources_target ON sources(target_key);
CREATE INDEX IF NOT EXISTS idx_scores_target ON scores(target_key);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Store:
    """Standing database: persistent search cache plus an accumulating record of
    every candidate connector, its citations, and per-run scores."""

    def __init__(self, path: str, *, cache_ttl_hours: int = 168, use_cache: bool = True):
        self.path = path
        self.cache_ttl_hours = cache_ttl_hours
        self.use_cache = use_cache
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ----- search cache (consumed by SearchEngine) -----
    def get_search(self, cache_key: str) -> Optional[list[SearchResult]]:
        if not self.use_cache:
            return None
        row = self.conn.execute(
            "SELECT created_at, payload FROM search_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if not row:
            return None
        try:
            created = datetime.fromisoformat(row["created_at"])
        except ValueError:
            return None
        if datetime.now() - created > timedelta(hours=self.cache_ttl_hours):
            return None
        try:
            payload = json.loads(row["payload"])
        except json.JSONDecodeError:
            return None
        return [SearchResult(**item) for item in payload]

    def put_search(self, cache_key: str, results: list[SearchResult]) -> None:
        if not self.use_cache:
            return
        payload = json.dumps([r.__dict__ for r in results])
        self.conn.execute(
            "INSERT OR REPLACE INTO search_cache (cache_key, created_at, payload) VALUES (?, ?, ?)",
            (cache_key, _now(), payload),
        )
        self.conn.commit()

    # ----- run + candidate persistence -----
    def start_run(self, *, target, target_key, context, location, industry, period,
                  providers, extractor) -> int:
        cursor = self.conn.execute(
            "INSERT INTO runs (target, target_key, context, location, industry, period, "
            "providers, extractor, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (target, target_key, context, location or "", industry or "", period or "",
             ",".join(providers), extractor, _now()),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def record(self, run_id: int, target_key: str, scored: list[ScoredCandidate]) -> None:
        now = _now()
        for item in scored:
            cand = item.candidate
            primary_signal = best_signal(cand.signal_categories)
            self.conn.execute(
                "INSERT INTO candidates (name_key, name, first_seen, last_seen) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(name_key) DO UPDATE SET last_seen=excluded.last_seen, name=excluded.name",
                (cand.name_key, cand.name, now, now),
            )
            self.conn.execute(
                "INSERT INTO connections (candidate_key, target_key, signal_category, explanation, "
                "best_confidence, first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(candidate_key, target_key) DO UPDATE SET "
                "last_seen=excluded.last_seen, explanation=excluded.explanation, "
                "signal_category=excluded.signal_category, "
                "best_confidence=MAX(connections.best_confidence, excluded.best_confidence)",
                (cand.name_key, target_key, primary_signal,
                 cand.explanation, cand.extraction_confidence, now, now),
            )
            for source in cand.sources:
                self.conn.execute(
                    "INSERT OR IGNORE INTO sources (candidate_key, target_key, url, domain, title, "
                    "snippet, published_date, provider, query, signal_category, run_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (cand.name_key, target_key, source.url, source.domain, source.title,
                     source.snippet, source.published_date, source.provider, source.query,
                     source.signal_category, run_id, now),
                )
            self.conn.execute(
                "INSERT OR REPLACE INTO scores (run_id, candidate_key, target_key, score, tier, "
                "in_network, degree, via, distinct_domains, most_recent_date, stale_only, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, cand.name_key, target_key, item.score, item.tier,
                 1 if item.in_network else 0, item.degree, item.via, item.distinct_domains,
                 item.most_recent_date, 1 if item.stale_only else 0, now),
            )
        self.conn.execute("UPDATE runs SET n_candidates = ? WHERE id = ?", (len(scored), run_id))
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
