# Connection Finder

A warm-introduction path finder for **Artemis**. Give it a target person and a
short disambiguation context, and it searches public web sources for other named
individuals who co-occur with the target — surfacing people you could realistically
ask for a warm intro, each with a confidence tier, signal category, source
citations, and a flag for whether they're already in your LinkedIn network.

Standard-library only. No `pip install` required to run.

## Quick start

```bash
cp .env.example .env        # add at least one search provider key
python run.py "Bill Gates" --context "Microsoft"
```

Or as a module:

```bash
python -m connection_finder.cli "Enrique Linares" --context "Plus Partners" \
  --connections Connections.csv \
  --out gates.md --json gates.json
```

Interactive (prompts for target + context):

```bash
python run.py
```

## How it works (the pipeline)

1. **Query construction** — builds a disambiguated batch across five signal
   categories, always quoting `target` + `context`:
   professional co-occurrence · institutional affiliation · public social proof ·
   joint appearances · broad/incidental.
2. **Search** — a single `search(query)` abstraction fans each query out across
   every configured provider (Brave / Google CSE / Bing), merges and dedupes the
   results, and caches them in the standing DB. Rate-limited providers are backed
   off and disabled for the rest of the run; one provider failing never aborts the run.
3. **Content extraction** — fetches and cleans the full text of the top results
   per query (not just snippets) and pulls a publication date from meta tags /
   JSON-LD / the URL.
4. **Entity extraction** — Gemini reads the evidence and returns people connected
   to the target, each with a citation URL, signal category, and confidence. With
   no `GEMINI_API_KEY`, a **precision-first** no-LLM heuristic extractor takes over:
   it only emits a name when that name sits close to a mention of the target *and*
   to a role/relationship word (founder, investor, colleague, classmate, ...), which
   keeps product/place/website-chrome names out. (This trades recall for precision;
   Gemini recovers the recall.)
4b. **Photo analysis** (optional, `--analyze-photos`, needs Gemini) — for images
   that have **no caption**, a vision pass reads names that are visibly in the
   image (name badges, event banners, nameplates, lower-thirds) and identifies
   clearly-recognizable public figures pictured with the target. It never guesses
   the identity of unrecognized/private faces and never includes minors. Captioned
   images are skipped — their names already flow through text extraction.
5. **Dedupe + scoring** — candidates are merged by normalized name and scored on a
   weighted system: signal category (professional/institutional highest, incidental
   lowest), multi-source corroboration (distinct domains), recency (sources older
   than ~15 years with no recent corroboration are penalized), and extraction
   confidence. Each gets a **high / medium / low** tier.
6. **Network cross-reference** — every candidate is matched against your LinkedIn
   `Connections.csv` (and an optional 2nd-degree map) to flag whether you're already
   1 or 2 hops away, which boosts actionable connectors.

Everything is written to a **standing SQLite database** (`connection_finder.sqlite3`
by default) so candidates, citations, and per-run scores accumulate over time.

## Configuration

Keys live in `.env` (see `.env.example`). The tool degrades gracefully — it uses
whatever subset of these is present and needs at least one search provider:

| Variable | Purpose |
|---|---|
| `BRAVE_API_KEY` | Brave Search API (recommended default) |
| `GOOGLE_CSE_ID` + `GOOGLE_CSE_KEY` | Google Programmable Search Engine |
| `BING_API_KEY` | Bing Web Search v7 (being retired by Microsoft) |
| `GEMINI_API_KEY` | High-quality extraction (optional) |

`BRAVE_SEARCH_API_KEY` is accepted as an alias for `BRAVE_API_KEY`.

## Useful flags

```text
--context, -c        disambiguation string (required)
--location / --industry / --period   narrow toward a specific scene or era
--connections PATH   your LinkedIn Connections.csv (enables in-network flags)
--second-degree PATH JSON {connection_name: [their_connections]} for 2nd-degree
--providers LIST     restrict to e.g. brave,google_cse
--max-results N      results per query (default 6)
--max-pages N        results to fetch full text for, per query (default 3)
--max-queries N      cap total queries (0 = full batch)
--no-fetch           snippets only — faster and cheaper
--analyze-photos     vision-analyze uncaptioned photos for named people (needs Gemini)
--max-photos N       cap photos analyzed per run (default 4)
--stale-years N      penalize sole evidence older than N years (default 15)
--no-cache           bypass the persistent search cache
--out FILE.md        write a Markdown report
--json FILE.json     write structured JSON
--db PATH            standing SQLite DB location
```

## Output

Console table plus optional Markdown / JSON. Each candidate carries: name,
one-line connection-to-target explanation, confidence tier, primary signal
category, source citations (URL + date), and in-network degree-of-separation.

## Safety constraints

- Public web sources only. Every claim must carry a citation URL.
- Rejects family inference, minors, private addresses/contact details, leaked
  data, and hidden/private accounts.
- Treats results as **possible** public overlaps, not confirmed relationships,
  until you verify them.
