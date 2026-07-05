"""Offline smoke test — exercises the pipeline without network access or keys.

Run from the project root:
    python tests/smoke_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connection_finder.config import ConfigError, Settings
from connection_finder.dates import extract_date_from_html, to_iso_date, years_old
from connection_finder.extract import heuristic_extract, photo_candidates
from connection_finder.images import ImageRef, _mime_for, extract_images
from connection_finder.models import SearchResult
from connection_finder.network import build_index
from connection_finder.pipeline import RunResult
from connection_finder.query import build_queries
from connection_finder.report import render_console, render_markdown
from connection_finder.safety import is_sensitive
from connection_finder.score import score_and_rank
from connection_finder.search.base import SearchProvider
from connection_finder.search.engine import SearchEngine, _merge
from connection_finder.store import Store
from connection_finder.util import registrable_domain
from connection_finder.http_util import RateLimitError

PASS = 0
FAIL = 0


def check(label: str, condition: bool) -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ok  {label}")
    else:
        FAIL += 1
        print(f"FAIL  {label}")


# --- 1. Query construction ------------------------------------------------- #
def test_queries():
    specs = build_queries("Bill Gates", "Microsoft", period="2010s")
    texts = [s.text for s in specs]
    cats = {s.signal_category for s in specs}
    check("queries quote target+context", all('"Bill Gates"' in t and '"Microsoft"' in t for t in texts))
    check("queries cover 5 categories", {
        "professional_co_occurrence", "institutional_affiliation",
        "social_proof", "joint_appearance", "incidental",
    } <= cats)
    check("queries target family + close friends", {"family", "close_friend"} <= cats)
    check("family query present", any(w in t.lower() for t in texts for w in ("wife", "spouse", "sibling", "family")))
    check("niche/social site query present", any("site:facebook.com" in t or "site:instagram.com" in t for t in texts))
    check("queries are deduped", len(texts) == len(set(texts)))
    check("period probe present", any("2010s" in t for t in texts))


# --- 2. Dates -------------------------------------------------------------- #
def test_dates():
    check("ISO passthrough", to_iso_date("2019-04-12T08:00:00Z") == "2019-04-12")
    check("text date parse", to_iso_date("May 3, 2018") == "2018-05-03")
    check("bare year", to_iso_date("class of 2004") == "2004-01-01")
    html = '<meta property="article:published_time" content="2016-07-01T10:00:00Z">'
    check("html meta date", extract_date_from_html(html, "") == "2016-07-01")
    check("url date", extract_date_from_html("", "https://x.com/2012/06/05/story") == "2012-06-05")
    age = years_old("2000-01-01")
    check("years_old computes", age is not None and age > 20)


# --- 3. Engine merge + provider disable ------------------------------------ #
class FakeProvider(SearchProvider):
    def __init__(self, name, results):
        self.name = name
        self._results = results

    def search(self, query, count):
        return list(self._results)


class BoomProvider(SearchProvider):
    name = "boom"

    def search(self, query, count):
        raise RateLimitError("boom: rate limited")


def _sr(provider, url, title="T", date=""):
    return SearchResult(query="q", signal_category="", provider=provider, title=title,
                        url=url, snippet="s", rank=1, published_date=date)


def test_engine():
    a = FakeProvider("a", [_sr("a", "https://site.com/x", date="2020-01-01"), _sr("a", "https://other.com/y")])
    b = FakeProvider("b", [_sr("b", "https://site.com/x/"), _sr("b", "https://third.com/z")])
    merged = _merge([a._results, b._results])
    urls = {m.url for m in merged}
    check("merge dedupes by url", len(merged) == 3)
    check("merge keeps distinct hosts", {"https://site.com/x", "https://other.com/y", "https://third.com/z"} == urls)

    engine = SearchEngine([a, b, BoomProvider()], cache=None, verbose=False)
    out = engine.search("q", 5, "incidental")
    check("engine fans out + tags category", out and all(r.signal_category == "incidental" for r in out))
    check("engine disables rate-limited provider", "boom" in engine._disabled)


# --- 4. Heuristic extraction ----------------------------------------------- #
def test_extraction():
    results = [
        SearchResult(query="q", signal_category="professional_co_occurrence", provider="a",
                     title="Paul Allen and Bill Gates founded Microsoft",
                     url="https://news.com/a", snippet="Paul Allen co-founded the company with Bill Gates.",
                     published_date="2021-01-01",
                     page_text="Paul Allen co-founded Microsoft with Bill Gates. Steve Ballmer later joined."),
        SearchResult(query="q", signal_category="professional_co_occurrence", provider="b",
                     title="Ballmer profile", url="https://blog.org/b",
                     snippet="Steve Ballmer worked closely with Bill Gates for decades.",
                     published_date="2020-06-01",
                     page_text="Steve Ballmer worked closely with Bill Gates."),
    ]
    raws = heuristic_extract("Bill Gates", results)
    names = {r.name for r in raws}
    check("heuristic finds co-occurring people", "Paul Allen" in names and "Steve Ballmer" in names)
    check("heuristic excludes the target", "Bill Gates" not in names)
    check("heuristic attaches citations", all(r.citation_url for r in raws))
    return raws, results


# --- 5. Scoring + network + report ----------------------------------------- #
def test_scoring_and_report():
    raws, results = test_extraction()
    results_by_url = {r.url: r for r in results}

    # Fake network: Steve Ballmer is a direct connection.
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "conn.csv")
        with open(csv_path, "w", encoding="utf-8") as fh:
            fh.write("First Name,Last Name,URL,Company,Position\n")
            fh.write("Steve,Ballmer,https://linkedin.com/in/sb,LA Clippers,Owner\n")
        index = build_index(csv_path, "", verbose=False)
        check("network index loads", index.size[0] == 1)

        scored, _removed = score_and_rank(raws, results_by_url, network=index, stale_years=15, recent_years=5)
        check("scoring returns candidates", len(scored) >= 2)
        check("scores are sorted desc", all(
            scored[i].score >= scored[i + 1].score for i in range(len(scored) - 1)))
        ballmer = next((s for s in scored if s.candidate.name_key == "steve ballmer"), None)
        check("in-network flagged degree 1", ballmer is not None and ballmer.in_network and ballmer.degree == 1)
        check("tiers assigned", all(s.tier in ("high", "medium", "low") for s in scored))

        result = RunResult(target="Bill Gates", context="Microsoft", scored=scored,
                           queries_run=13, results_seen=2, providers=["fake"], extractor="heuristic")
        console = render_console(result)
        md = render_markdown(result)
        check("console renders", "Bill Gates" in console and "Candidate" in console)
        check("markdown renders with citations", "Sources" in md and "https://" in md)


# --- 6. Store round-trip --------------------------------------------------- #
def test_store():
    raws, results = test_extraction()
    results_by_url = {r.url: r for r in results}
    scored, _removed = score_and_rank(raws, results_by_url, network=None)
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "db.sqlite3")
        store = Store(db)
        run_id = store.start_run(target="Bill Gates", target_key="bill gates", context="Microsoft",
                                 location=None, industry=None, period=None,
                                 providers=["brave"], extractor="heuristic")
        store.record(run_id, "bill gates", scored)
        n_cand = store.conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        n_src = store.conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        n_score = store.conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        check("store persists candidates", n_cand >= 2)
        check("store persists sources", n_src >= 2)
        check("store persists scores", n_score >= 2)
        # Second run accumulates without duplicating sources (same URLs).
        run2 = store.start_run(target="Bill Gates", target_key="bill gates", context="Microsoft",
                               location=None, industry=None, period=None,
                               providers=["brave"], extractor="heuristic")
        store.record(run2, "bill gates", scored)
        n_src2 = store.conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        check("re-running does not duplicate sources", n_src2 == n_src)
        store.close()


# --- 7. Graceful degradation ----------------------------------------------- #
def test_config():
    empty = Settings()  # no keys at all
    check("no providers when no keys", empty.available_providers() == [])
    raised = False
    try:
        empty.validate_for_search()
    except ConfigError:
        raised = True
    check("validate raises clear error with no providers", raised)

    one = Settings(brave_api_key="x")
    check("brave-only is available", one.available_providers() == ["brave"])
    check("brave-only validates", _no_raise(one.validate_for_search))
    cse = Settings(google_cse_id="id", google_cse_key="key")
    check("google needs both id+key", cse.available_providers() == ["google_cse"])
    check("google missing key -> unavailable", Settings(google_cse_id="id").available_providers() == [])


def _no_raise(fn) -> bool:
    try:
        fn()
        return True
    except Exception:
        return False


# --- 8. Regression tests for the review fixes ------------------------------ #
class _DictCache:
    def __init__(self):
        self.store = {}

    def get_search(self, key):
        if key not in self.store:
            return None
        return [SearchResult(**r.__dict__) for r in self.store[key]]

    def put_search(self, key, results):
        self.store[key] = list(results)


def test_fixes():
    # Date days 29-31 are no longer clamped to 28.
    check("May 31 not clamped", to_iso_date("May 31, 2018") == "2018-05-31")
    check("30 Jan not clamped", to_iso_date("30 January 2020") == "2020-01-30")
    check("impossible day -> year", to_iso_date("February 30, 2019") == "2019-01-01")
    check("month-year normalizes", to_iso_date("May 2018") == "2018-05-01")

    # Port no longer splits a single domain in two.
    check("port collapses to one domain",
          registrable_domain("https://example.com/a") == registrable_domain("https://example.com:8443/b"))

    # Safety: adult family is now ALLOWED (it is the tool's purpose)...
    check("allows adult 'their son Mark'", not is_sensitive("their son Mark joined the firm"))
    check("allows possessive 's daughter (adult)", not is_sensitive("at John's daughter's wedding"))
    check("allows 'his wife'", not is_sensitive("his wife Melinda co-chairs the foundation"))
    check("allows 'baby brother' adult idiom", not is_sensitive("his baby brother, now 40, runs the firm"))
    check("allows surname 'Baby'", not is_sensitive("longtime colleague", "Lisa Baby"))
    # ...but minors are still blocked, including plural / alternate phrasings.
    check("blocks 'his teenage daughter'", is_sensitive("introduced his teenage daughter Jane"))
    check("blocks 'teenagers' plural", is_sensitive("their two teenagers attended"))
    check("blocks 'teens'", is_sensitive("the teens joined them"))
    check("blocks age-of-minor", is_sensitive("pictured with his 16-year-old"))
    check("blocks 'N years old' plural", is_sensitive("his son Jack, 8 years old"))
    check("blocks 'aged 12'", is_sensitive("his daughter, aged 12"))
    check("blocks 'minor child'", is_sensitive("appeared with a minor child"))
    check("blocks 'minor son'", is_sensitive("pictured with his minor son"))
    check("blocks 'toddlers' plural", is_sensitive("photographed with their toddlers"))
    check("blocks 'baby girl'", is_sensitive("welcomed a baby girl"))
    check("blocks '4th grader'", is_sensitive("a 4th grader at the local school"))
    # ...and neutral business phrasing is unaffected.
    check("allows 'sister company'", not is_sensitive("launched a sister company with Jane Doe"))
    check("allows 'parent organization'", not is_sensitive("the parent organization of Acme"))
    # Private/contact data is always blocked, singular and plural.
    check("blocks home address", is_sensitive("published his home address online"))
    check("blocks phone number", is_sensitive("leaked her phone number"))
    check("blocks 'phone numbers' plural", is_sensitive("his phone numbers are listed"))
    check("blocks 'passports' plural", is_sensitive("their passports were shown"))

    # Heuristic safety inspects context near LATER name occurrences too — a minor
    # marker anywhere near a name still drops it.
    long_gap = "x " * 120
    page = f"Jane Doe, a colleague of Bill Gates, spoke. {long_gap} the 15-year-old Jane Doe attended."
    res = SearchResult(query="q", signal_category="incidental", provider="a",
                       title="t", url="https://news.com/x", snippet="", page_text=page)
    names = {r.name for r in heuristic_extract("Bill Gates", [res])}
    check("heuristic drops name flagged (minor) near a later occurrence", "Jane Doe" not in names)

    # Precision gate: a name needs a person-signal near a target mention.
    noise = SearchResult(query="q", signal_category="incidental", provider="a", title="t",
                         url="https://n.com/y", snippet="",
                         page_text="Bill Gates used Windows Vista in Silicon Valley last year.")
    noise_names = {r.name for r in heuristic_extract("Bill Gates", [noise])}
    check("precision: product/place noise dropped", noise_names == set())

    real = SearchResult(query="q", signal_category="professional_co_occurrence", provider="a",
                        title="t", url="https://n.com/z", snippet="",
                        page_text="Nathan Myhrvold, a longtime colleague of Bill Gates, led research.")
    real_names = {r.name for r in heuristic_extract("Bill Gates", [real])}
    check("precision: real signalled person kept", "Nathan Myhrvold" in real_names)

    # A source that never mentions the target yields nothing.
    offtopic = SearchResult(query="q", signal_category="incidental", provider="a", title="t",
                            url="https://n.com/o", snippet="",
                            page_text="Jane Roe co-founded Acme with John Poe.")
    check("off-topic source (no target) dropped",
          heuristic_extract("Bill Gates", [offtopic]) == [])

    # Engine never caches a partial (degraded) result under a complete key.
    cache = _DictCache()
    good = FakeProvider("a", [_sr("a", "https://s.com/x")])
    engine = SearchEngine([good, BoomProvider()], cache=cache, verbose=False)
    engine.search("q", 5, "incidental")
    check("partial run not cached", cache.store == {})
    # Now boom is disabled; the next call is complete for the active set and caches.
    engine.search("q2", 5, "incidental")
    check("complete run cached under active-only key", "a|5|q2" in cache.store)

    # Surname-only candidate folds into the unique full-name candidate.
    from connection_finder.extract import RawCandidate
    from connection_finder.score import merge_candidates
    raws2 = [
        RawCandidate(name="Satya Nadella", explanation="e", signal_category="professional_co_occurrence",
                     confidence=0.8, citation_url="https://a.com/1"),
        RawCandidate(name="Nadella", explanation="e", signal_category="professional_co_occurrence",
                     confidence=0.7, citation_url="https://b.com/2"),
    ]
    merged = merge_candidates(raws2, {})
    keys = {c.name_key for c in merged}
    check("surname folds into full name", keys == {"satya nadella"})
    satya = merged[0]
    check("folded candidate keeps both sources", len(satya.sources) == 2)

    # Middle-name variants (same first + surname) collapse into the fullest name.
    raws3 = [
        RawCandidate(name="Melinda Gates", explanation="e1", signal_category="family",
                     confidence=0.7, citation_url="https://a.com/1"),
        RawCandidate(name="Melinda French Gates", explanation="e3", signal_category="family",
                     confidence=0.9, citation_url="https://c.com/3"),
    ]
    merged3 = merge_candidates(raws3, {})
    keys3 = {c.name_key for c in merged3}
    check("middle-name variant collapses to fullest name", keys3 == {"melinda french gates"})
    check("collapsed variant keeps both sources", len(merged3[0].sources) == 2)

    # Ambiguous shared first name is NOT merged (two different people).
    raws4 = [
        RawCandidate(name="Paul Allen", explanation="e", signal_category="close_friend",
                     confidence=0.8, citation_url="https://a.com/1"),
        RawCandidate(name="Paul Gilbert", explanation="e", signal_category="close_friend",
                     confidence=0.8, citation_url="https://b.com/2"),
    ]
    merged4 = merge_candidates(raws4, {})
    check("distinct people with shared first name stay separate", len(merged4) == 2)

    # Two different people sharing a first name must NOT be fused via a spurious
    # union name ("Michael Jordan" + "Michael Jackson" + "Michael Jordan Jackson").
    raws5 = [
        RawCandidate(name="Michael Jordan", explanation="e", signal_category="incidental",
                     confidence=0.8, citation_url="https://a.com/1"),
        RawCandidate(name="Michael Jackson", explanation="e", signal_category="incidental",
                     confidence=0.8, citation_url="https://b.com/2"),
        RawCandidate(name="Michael Jordan Jackson", explanation="e", signal_category="incidental",
                     confidence=0.8, citation_url="https://c.com/3"),
    ]
    merged5 = merge_candidates(raws5, {})
    keys5 = {c.name_key for c in merged5}
    check("distinct people not fused via spurious union (Jordan preserved)",
          "michael jordan" in keys5 and len(merged5) >= 2)

    # Individuals are NOT absorbed into a multi-person conjunction name.
    raws6 = [
        RawCandidate(name="Bill Gates", explanation="e", signal_category="professional_co_occurrence",
                     confidence=0.8, citation_url="https://a.com/1"),
        RawCandidate(name="Melinda Gates", explanation="e", signal_category="family",
                     confidence=0.8, citation_url="https://b.com/2"),
        RawCandidate(name="Bill and Melinda Gates", explanation="e", signal_category="family",
                     confidence=0.8, citation_url="https://c.com/3"),
    ]
    merged6 = merge_candidates(raws6, {})
    keys6 = {c.name_key for c in merged6}
    check("individuals not absorbed into conjunction",
          "bill gates" in keys6 and "melinda gates" in keys6)

    # Network approximate match is hedged, not asserted as a full 1st-degree.
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "c.csv")
        with open(csv_path, "w", encoding="utf-8") as fh:
            fh.write("First Name,Last Name\nJohn,Smith\n")
        index = build_index(csv_path, "", verbose=False)
        exact = index.lookup("John Smith")
        approx = index.lookup("John Aaron Smith")
        check("exact match not approximate", exact.in_network and not exact.approximate)
        check("first+last collision flagged approximate", approx.in_network and approx.approximate)


# --- 9. Photo / image discovery -------------------------------------------- #
_SAMPLE_HTML = """
<html><head>
<meta property="og:image" content="/hero.jpg">
</head><body>
<img src="https://cdn.site.com/logo.png" alt="Site Logo">
<img src="https://cdn.site.com/team-photo.jpg" alt="">
<figure>
  <img src="/photos/event.jpg" alt="">
  <figcaption>Bill Gates and Paul Allen at the launch</figcaption>
</figure>
<img src="https://cdn.site.com/icon-sprite.svg" alt="">
<img src="https://cdn.site.com/gala.jpg" alt="Annual gala dinner">
</body></html>
"""


def test_images():
    imgs = extract_images(_SAMPLE_HTML, "https://site.com/page")
    urls = {i.url for i in imgs}
    check("og:image captured + resolved", "https://site.com/hero.jpg" in urls)
    check("relative figure img resolved", "https://site.com/photos/event.jpg" in urls)
    check("logo filtered out", "https://cdn.site.com/logo.png" not in urls)
    check("svg/sprite filtered out", "https://cdn.site.com/icon-sprite.svg" not in urls)

    by_url = {i.url: i for i in imgs}
    event = by_url.get("https://site.com/photos/event.jpg")
    team = by_url.get("https://cdn.site.com/team-photo.jpg")
    gala = by_url.get("https://cdn.site.com/gala.jpg")
    check("figcaption attached", event is not None and "Bill Gates" in event.caption)
    check("figure image is captioned", event is not None and event.has_caption())
    check("empty-alt image is uncaptioned", team is not None and not team.has_caption())
    check("meaningful alt counts as caption", gala is not None and gala.has_caption())
    check("generic alt is not a caption", not ImageRef(url="x", alt="photo image").has_caption())

    # mime resolution
    check("mime from extension", _mime_for("https://x/y.jpg", "") == "image/jpeg")
    check("mime from content-type", _mime_for("https://x/y", "image/png; charset=x") == "image/png")
    check("no mime for non-image", _mime_for("https://x/y.svg", "") is None)

    # Photo analysis is a no-op without Gemini.
    check("photo_candidates no-op without gemini",
          photo_candidates("Bill Gates", "Microsoft", imgs, gemini=None, max_photos=4) == [])


# --- 10. Fame filtering (remove well-known people, keep niche) ------------- #
def test_fame():
    from connection_finder.extract import RawCandidate, _plausible_person_name
    from connection_finder.score import score_and_rank, _has_own_wikipedia, merge_candidates

    raws = [
        RawCandidate(name="Melinda French Gates", explanation="ex-wife", signal_category="family",
                     confidence=0.9, citation_url="https://a.com/1", prominence="household_name"),
        RawCandidate(name="Rory Gates", explanation="son", signal_category="family",
                     confidence=0.8, citation_url="https://b.com/2", prominence="private"),
        RawCandidate(name="Kristi Blake", explanation="sister", signal_category="family",
                     confidence=0.8, citation_url="https://c.com/3", prominence="niche"),
    ]
    kept, removed = score_and_rank(raws, {}, remove_famous=True, max_fame=0.6)
    kept_names = {s.candidate.name for s in kept}
    removed_names = {s.candidate.name for s in removed}
    check("household-name person removed", "Melinda French Gates" in removed_names)
    check("niche/private people kept", {"Rory Gates", "Kristi Blake"} <= kept_names)
    check("kept carry a fame score", all(0.0 <= s.fame <= 1.0 for s in kept))
    check("removed marked more famous than kept",
          all(r.fame >= 0.6 for r in removed) and all(k.fame < 0.6 for k in kept))

    # --keep-famous: nobody is removed.
    kept2, removed2 = score_and_rank(raws, {}, remove_famous=False)
    check("keep-famous keeps everyone", len(kept2) == 3 and removed2 == [])

    # A dedicated Wikipedia article marks someone famous even if rated 'private'.
    raws_wiki = [RawCandidate(name="Jane Q Roe", explanation="x", signal_category="incidental",
                 confidence=0.5, citation_url="https://en.wikipedia.org/wiki/Jane_Q_Roe",
                 prominence="private")]
    merged = merge_candidates(raws_wiki, {})
    check("own wikipedia article detected", _has_own_wikipedia(merged[0]))
    _, removed3 = score_and_rank(raws_wiki, {}, remove_famous=True, max_fame=0.6)
    check("own wikipedia article => filtered as famous",
          any(s.candidate.name == "Jane Q Roe" for s in removed3))

    # Free person-name backstop keeps people, drops headline/label junk.
    check("backstop keeps a real name", _plausible_person_name("Mary Jane Watson"))
    check("backstop keeps name with suffix", _plausible_person_name("William Henry Gates II"))
    check("backstop drops headline label", not _plausible_person_name("When To Use"))
    check("backstop drops single token", not _plausible_person_name("Melinda"))


def main() -> int:
    for test in (test_queries, test_dates, test_engine, test_scoring_and_report,
                 test_store, test_config, test_fixes, test_images, test_fame):
        print(f"\n== {test.__name__} ==")
        test()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
