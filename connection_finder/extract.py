from __future__ import annotations

import base64
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from urllib.parse import quote

from .dates import to_iso_date
from .http_util import HttpError, request_json
from .images import ImageRef, download_image
from .models import SIGNAL_CATEGORIES, SearchResult
from .safety import is_sensitive
from .util import looks_like_person_name, name_tokens, normalize_name, squeeze


@dataclass
class RawCandidate:
    """One extracted (person -> connected-to-target) claim from one source."""
    name: str
    explanation: str
    signal_category: str
    confidence: float
    citation_url: str
    evidence_quote: str = ""
    published_date: str = ""
    method: str = "heuristic"  # or "gemini"
    prominence: str = ""  # household_name | industry_known | niche | private


_VALID_CATEGORIES = set(SIGNAL_CATEGORIES)


def _clean_category(value, fallback: str) -> str:
    # Coerce first: Gemini occasionally returns a list/number even in JSON mode.
    value = str(value or "").strip().lower().replace(" ", "_")
    return value if value in _VALID_CATEGORIES else fallback


# --------------------------------------------------------------------------- #
# Heuristic extractor (no LLM required)
# --------------------------------------------------------------------------- #

_NAME_RE = re.compile(
    r"\b([A-Z][a-z]+(?:[''\-][A-Za-z]+)?(?:\s+(?:[A-Z]\.|[A-Z][a-z]+)){1,2})\b"
)

_NON_NAME_WORDS = {
    "the", "this", "that", "these", "those", "read", "more", "learn", "home",
    "about", "contact", "search", "menu", "sign", "login", "privacy", "terms",
    "policy", "cookie", "cookies", "follow", "share", "subscribe", "newsletter",
    "getting", "started", "click", "here", "view", "show", "hide", "next",
    "previous", "download", "update", "updated", "latest", "breaking", "related",
    "sponsored", "advertisement", "watch", "listen", "skip", "close", "submit",
    "reply", "comment", "comments", "section", "chapter", "overview", "summary",
    "introduction", "conclusion", "references", "copyright", "reserved",
    "anonymous", "unknown", "advertisement", "newsletter",
    "windows", "powerpoint", "outlook", "azure", "xbox", "copilot", "iphone",
    "ipad", "android", "chrome", "safari", "linux", "sharepoint", "onedrive",
    "dynamics", "cortana", "internet", "explorer",
}

_PERSON_SIGNALS = (
    # Family / personal ties (the tool's primary target).
    "wife", "husband", "spouse", "fiance", "fiancee", "partner", "married",
    "wedding", "brother", "sister", "sibling", "father", "mother", "parents",
    "son", "daughter", "family", "relative", "cousin", "nephew", "niece",
    # Friendship.
    "friend", "friends", "friendship", "befriended", "longtime", "childhood",
    "close", "confidant", "confidante", "godfather", "godmother",
    # Professional / institutional.
    "founder", "co-founder", "cofounder", "founded", "co-founded", "cofounded",
    "ceo", "cto", "cfo", "coo", "cmo", "president", "chairman", "chairwoman",
    "chairperson", "director", "executive", "chief", "chairs",
    "investor", "investors", "backer", "backed", "angel", "venture",
    "partners", "advisor", "adviser", "advisory", "board", "trustee",
    "colleague", "colleagues", "classmate", "classmates", "roommate", "teammate",
    "coworker", "co-worker", "mentor", "mentored", "protege", "mentee",
    "professor", "student", "students", "graduate", "graduated", "alumnus",
    "alumna", "alumni", "fellow", "fellowship", "phd",
    "coauthor", "co-author", "collaborator", "collaborated", "co-wrote",
    "employee", "hired", "recruited", "joined", "worked", "working", "alongside",
    "partnered", "teamed", "successor", "predecessor", "succeeded", "replaced",
    "interviewed", "panel", "panelist", "speaker", "keynote", "podcast", "guest",
    "host", "appointed", "spoke", "met",
)
_PERSON_SIGNAL_RE = re.compile(
    r"\b(" + "|".join(sorted(set(_PERSON_SIGNALS), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_TARGET_WINDOW = 320
_SIGNAL_WINDOW = 60


def _windows_around(text: str, name: str, width: int = 160, max_windows: int = 6) -> str:
    windows: list[str] = []
    start = 0
    while len(windows) < max_windows:
        idx = text.find(name, start)
        if idx < 0:
            break
        lo = max(0, idx - width // 2)
        windows.append(text[lo: idx + len(name) + width // 2])
        start = idx + len(name)
    if not windows:
        return squeeze(text[:width])
    return squeeze(" … ".join(windows))


def _all_positions(low_text: str, needle: str) -> list[int]:
    positions: list[int] = []
    needle = needle.lower()
    if not needle:
        return positions
    start = 0
    while True:
        idx = low_text.find(needle, start)
        if idx < 0:
            break
        positions.append(idx)
        start = idx + len(needle)
    return positions


def _has_non_name_word(name: str) -> bool:
    return any(tok.lower() in _NON_NAME_WORDS for tok in name.split())


# Function/headline words that never appear as a token in a real person's name.
# Lets the free person-name backstop reject obvious labels ("When To Use") so we
# can skip the paid LLM verification pass by default.
_NON_PERSON_TOKENS = {
    "when", "to", "use", "using", "used", "how", "why", "what", "which", "who",
    "whom", "where", "get", "getting", "best", "top", "guide", "tips", "vs",
    "versus", "is", "are", "was", "were", "your", "you", "our", "about",
    "contact", "read", "learn", "sign", "click", "here", "view", "show", "hide",
    "faq", "review", "overview", "for", "with", "from", "into", "list",
    "interview", "meeting", "event", "summit", "episode", "story", "report",
    "profile", "news",
}
_NAME_PARTICLES = {
    "van", "von", "de", "del", "della", "di", "da", "la", "le", "bin", "al",
    "der", "ten", "ter", "du", "dos", "das", "mac", "mc", "o",
}


def _plausible_person_name(name: str) -> bool:
    """Free, lenient check that an LLM-returned string is really a person's name.

    Looser than looks_like_person_name (allows middle names and suffixes), it
    only kills obvious non-people — headlines, labels, product names — so the
    paid verification pass can stay off by default.
    """
    tokens = name.strip().split()
    if not (2 <= len(tokens) <= 5):
        return False
    key = name_tokens(name)
    if len(key) < 2:
        return False
    if any(tok in _NON_NAME_WORDS or tok in _NON_PERSON_TOKENS for tok in key):
        return False
    caps = 0
    for index, tok in enumerate(tokens):
        if tok[:1].isupper():
            caps += 1
        elif tok.lower() in _NAME_PARTICLES and index > 0:
            continue  # lowercase particle mid-name ("van", "de") is fine
        else:
            return False
    return caps >= 2


def heuristic_extract(target_name: str, results: list[SearchResult]) -> list[RawCandidate]:
    """Precision-first extraction without an LLM.

    A name is emitted only when it (1) looks like a person, (2) sits within
    ``_TARGET_WINDOW`` chars of a mention of the target, and (3) has a
    person-signal word within ``_SIGNAL_WINDOW`` chars.
    """
    target_key = normalize_name(target_name)
    target_token_set = set(name_tokens(target_name))
    surname = (target_name.split() or [""])[-1]
    # Pre-compile word-boundary pattern for surname fallback.
    surname_re = re.compile(rf"\b{re.escape(surname.lower())}\b") if len(surname) >= 3 else None
    out: list[RawCandidate] = []
    for result in results:
        text = " ".join(filter(None, [result.title, result.snippet, result.page_text]))
        if not text:
            continue
        low = text.lower()
        # The source must mention the target (full name or surname, word-bounded).
        target_positions = _all_positions(low, target_name.lower())
        if not target_positions and surname_re and surname.lower() not in _NON_NAME_WORDS:
            target_positions = [m.start() for m in surname_re.finditer(low)]
        if not target_positions:
            continue

        seen_in_result: set[str] = set()
        max_names_per_source = 12
        for match in _NAME_RE.finditer(text):
            if len(seen_in_result) >= max_names_per_source:
                break
            name = match.group(1).strip()
            if not looks_like_person_name(name) or _has_non_name_word(name):
                continue
            key = normalize_name(name)
            if not key or key == target_key or key in seen_in_result:
                continue
            if set(name_tokens(name)).issubset(target_token_set):
                continue

            idx = match.start()
            if min(abs(idx - pos) for pos in target_positions) > _TARGET_WINDOW:
                continue
            lo = max(0, idx - _SIGNAL_WINDOW)
            hi = idx + len(name) + _SIGNAL_WINDOW
            signal_match = _PERSON_SIGNAL_RE.search(text, lo, hi)
            if not signal_match:
                continue

            evidence = _windows_around(text, name)
            if is_sensitive(evidence, name):
                continue

            seen_in_result.add(key)
            signal_word = signal_match.group(1).lower()
            distance = abs(signal_match.start() - idx)
            confidence = 0.42 if distance <= 30 else 0.34
            out.append(RawCandidate(
                name=name,
                explanation=f"Appears near '{signal_word}' in a source about {target_name}.",
                signal_category=result.signal_category or "incidental",
                confidence=confidence,
                citation_url=result.url,
                evidence_quote=evidence[:280],
                published_date=result.published_date,
                method="heuristic",
            ))
    return out


# --------------------------------------------------------------------------- #
# Gemini extractor
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = f"""
You find the CLOSE PERSONAL CIRCLE of a TARGET person — their family, close
friends, and closest personal/professional associates — to help arrange a warm
introduction. You read web search evidence and return ONLY JSON.

Return real, specifically-named people OTHER than the target who, ACCORDING TO
THE EVIDENCE, have a genuine documented connection to THIS EXACT target person.
Classify each into exactly one signal_category:

{", ".join(SIGNAL_CATEGORIES)}

Definitions (prioritise family and close_friend — they are what matters most):
- family: spouse/partner, sibling, parent, adult child, or other relative.
- close_friend: a close, longtime, or childhood friend; a personal confidant.
- professional_co_occurrence: co-founder, close colleague, board member,
  advisor, investor/investee, business partner.
- institutional_affiliation: same school/university, alumni, fellowship,
  classmate, roommate, lab, accelerator cohort.
- social_proof: named as a mentor, "early backer", "first believed in", public
  endorsement.
- joint_appearance: shared panel, conference, podcast, interview, event.
- incidental: any other genuine, specific personal connection not above.

For EACH person also rate their public prominence, so well-known people can be
filtered out (the user wants NICHE, harder-to-reach connections, not celebrities):
- household_name: globally famous — most people worldwide would recognize the
  name (e.g. a top tech CEO, a head of state, an A-list celebrity).
- industry_known: well known within their field, or clearly has their own
  Wikipedia article, but not a global household name.
- niche: a real public footprint (some articles, a company bio) but not widely
  known.
- private: an ordinary person with a minimal public profile.

RELEVANCE IS EVERYTHING. Only return a person if the evidence explicitly ties
THEM to the TARGET by name. Reject and DO NOT return:
- Anyone who merely appears in the same article, list, or webpage without a
  stated relationship to the target.
- Public figures, executives, or celebrities named only as examples, comparisons,
  or background — unless the evidence states a real personal/professional tie.
- Anything that is not a real person's name (headlines, product names, company
  names, section labels, navigation text, e.g. "Fidelity Interview", "when to
  use", "Getting Started", "Read More").

Other strict rules:
- Use ONLY the provided evidence. Every person MUST include a citation_url that
  is one of the evidence URLs.
- DO include adult family (spouse, siblings, parents, adult children). Do NOT
  include anyone who is or appears to be a minor (under 18).
- Use each person's FULL name (first and last) when the evidence provides it;
  never return a bare surname on its own.
- Do NOT use private addresses, contact details, leaked data, or medical info.
- If a connection is weak or speculative, lower confidence instead of inventing
  detail. Never fabricate a name or URL. Prefer FEWER, certain people.
- confidence is 0.0-1.0 reflecting how strongly the evidence supports a real,
  specific connection to the target.

JSON schema (no markdown, no commentary):
{{
  "people": [
    {{
      "name": "Full Name",
      "relationship": "one concise line stating exactly how they connect to the target",
      "signal_category": "family",
      "prominence": "household_name | industry_known | niche | private",
      "confidence": 0.0,
      "citation_url": "https://... (must be one of the evidence URLs)",
      "evidence_quote": "short verbatim quote from the evidence that names both people",
      "published_date": "YYYY-MM-DD or empty string"
    }}
  ]
}}
""".strip()

# Verification prompt — a strict relevance gate applied after extraction. Its
# whole job is to kill irrelevant and non-person entries before they reach the
# user, so results are ONLY real people genuinely connected to THIS target.
_VERIFY_SYSTEM_PROMPT = """
You are a strict relevance judge for a warm-introduction tool whose goal is to
map a TARGET person's CLOSE circle (family, friends, close associates).

You receive candidate entries (a NAME plus the evidence it came from). For each,
decide: is this a REAL, specific person who — per the evidence — has a GENUINE
documented connection to THIS EXACT target? Be skeptical. Default to REJECT when
unsure.

REJECT (do not include) any entry that is:
- Not a real person's name: phrases, headlines, product names, company names,
  event/section titles, navigation text (e.g. "when to use", "Fidelity
  Interview", "Getting Started", "Read More", "Privacy Policy").
- A real person with NO stated, specific relationship to the target — someone
  who merely co-appears in the same article, list, or search result.
- A public figure / celebrity / executive named only as an example, comparison,
  or background mention, with no actual tie to the target.
- Too vague, generic, or unverifiable to confirm a connection.

ACCEPT only real, specifically-named individuals (first + last name, or a
widely-known mononym) whom the evidence ties directly to the target — as family,
friend, or a documented professional/social connection.

Return ONLY JSON — no markdown, no commentary. Echo the accepted names EXACTLY
as given:
{"valid_names": ["Full Name 1", "Full Name 2"]}

If none pass, return: {"valid_names": []}
""".strip()


def _word_pattern(*terms: str):
    """Compile a WORD-BOUNDED, whitespace-flexible regex for the given terms.

    Word boundaries matter: a raw substring test makes the surname 'Cox' match
    'Coxsackievirus', 'Sun' match 'sunset', 'Day' match 'everyday' — which would
    keep off-topic pages and anchor focus on the wrong text. ``\\b`` around each
    term (matching the heuristic path) prevents that.
    """
    parts = []
    for term in sorted({t for t in terms if t}, key=len, reverse=True):
        words = term.split()
        if not words:
            continue
        parts.append(r"\b" + r"\s+".join(re.escape(w) for w in words) + r"\b")
    if not parts:
        return None
    return re.compile("|".join(parts), re.IGNORECASE)


def _target_patterns(target_name: str):
    """Return (full_name_pattern, broad_pattern).

    full_name_pattern matches only the exact full name; broad_pattern also
    matches the distinctive surname. Both are word-bounded.
    """
    full = target_name.strip()
    tokens = full.split()
    surname = tokens[-1].lower() if len(tokens) >= 2 else ""
    full_pat = _word_pattern(full)
    if len(surname) >= 3 and surname not in _NON_NAME_WORDS:
        broad_pat = _word_pattern(full, surname)
    else:
        broad_pat = full_pat
    return full_pat, broad_pat


def _mentions_target(text: str, pattern) -> bool:
    return bool(pattern and text and pattern.search(text))


def _focus_on_target(text: str, full_pat, broad_pat, *, width: int = 600,
                     max_windows: int = 8, cap: int = 4000) -> str:
    """Return only the regions of ``text`` around mentions of the target.

    Feeding Gemini the vicinity of the target — instead of the whole page —
    keeps extraction anchored to the person of interest and stops unrelated page
    sections (sidebars, related-article widgets, other people's bios) from
    producing bogus candidates. This is the core relevance defence.

    Exact full-name mentions are prioritised so that a page mostly about a
    different same-surname person can never crowd the target's own region out of
    the window budget.
    """
    if not text or full_pat is None:
        return ""

    def spans_for(pattern):
        out = []
        if pattern is None:
            return out
        for m in pattern.finditer(text):
            out.append((max(0, m.start() - width // 2), m.end() + width // 2))
        return out

    full_spans = spans_for(full_pat)
    broad_spans = spans_for(broad_pat) if broad_pat is not full_pat else []
    # Full-name windows first (never dropped), then surname windows fill the rest.
    prioritized = full_spans + broad_spans
    if not prioritized:
        return ""
    prioritized = prioritized[:max_windows * 3]

    prioritized.sort()
    merged: list[list[int]] = [list(prioritized[0])]
    for lo, hi in prioritized[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            if len(merged) >= max_windows:
                continue
            merged.append([lo, hi])
    focused = " … ".join(text[lo:hi] for lo, hi in merged)
    return squeeze(focused)[:cap]


def _relevant_results(target_name: str, results: list[SearchResult]) -> list[SearchResult]:
    """Keep only results whose title/snippet/page-text actually mention the
    target — off-topic search hits produce off-topic (irrelevant) people."""
    _, broad_pat = _target_patterns(target_name)
    kept: list[SearchResult] = []
    for r in results:
        blob = " ".join(filter(None, [r.title, r.snippet, r.page_text]))
        if _mentions_target(blob, broad_pat):
            kept.append(r)
    return kept


def _evidence_prompt(target_name: str, context: str, results: list[SearchResult]) -> str:
    full_pat, broad_pat = _target_patterns(target_name)
    blocks = []
    index = 0
    for result in results:
        # Prefer text focused around the target; fall back to the snippet.
        focused = _focus_on_target(result.page_text or "", full_pat, broad_pat)
        if not focused:
            snippet = result.snippet or ""
            focused = snippet if _mentions_target(snippet, broad_pat) else ""
        if not focused and not _mentions_target(result.title or "", broad_pat):
            continue  # nothing in this source actually discusses the target
        index += 1
        blocks.append("\n".join([
            f"Evidence {index}",
            f"Title: {result.title}",
            f"URL: {result.url}",
            f"Date: {result.published_date or 'unknown'}",
            f"Signal category of the query that found this: {result.signal_category}",
            f"Snippet: {result.snippet}",
            f"Text near the target: {focused}",
        ]))
    return (
        f"TARGET: {target_name}\n"
        f"CONTEXT (disambiguation): {context}\n\n"
        "From the evidence below, extract people who have a genuine documented "
        "connection to the TARGET (prioritise family and close friends). Only "
        "return a person if the evidence explicitly ties them to the target.\n\n"
        + "\n\n".join(blocks)
    )


# --------------------------------------------------------------------------- #
# Vision extraction
# --------------------------------------------------------------------------- #

_VISION_SYSTEM_PROMPT = """
You analyze a PHOTOGRAPH to help find warm-introduction connections to a TARGET
person. The image had NO caption, so its context is unknown. Return ONLY JSON.

You may report a person ONLY IF one of these is clearly true:
1. visible_text: their name is LITERALLY VISIBLE as text in the image — a name
   badge, name tag, nameplate, event banner, conference sign, slide, jersey, or
   a caption bar / lower-third burned into the image. Transcribe exactly.
2. recognized_public_figure: they are a widely-recognized public figure (e.g. a
   head of state, famous executive, well-known celebrity) whom you can identify
   with HIGH confidence and no ambiguity.

Hard rules — follow every one:
- DO NOT guess, infer, or fabricate the identity of any ordinary, private, or
  unrecognized person from their face, clothing, or appearance.
- DO NOT include anyone who appears to be a minor (under 18). Set appears_adult
  to false for anyone who could be under 18.
- DO NOT infer family relationships.
- Prefer returning FEWER, certain people over many guesses. If unsure, return
  an empty list.

JSON schema (no markdown, no commentary):
{
  "people": [
    {
      "name": "Full Name",
      "basis": "visible_text" | "recognized_public_figure",
      "appears_adult": true,
      "confidence": 0.0,
      "what_you_see": "short description of the text or figure this is based on"
    }
  ]
}
""".strip()


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
        # Guarantee a dict — the model occasionally returns a bare list.
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


class GeminiExtractor:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", *, retries: int = 4,
                 allow_insecure_ssl: bool = False):
        self.api_key = api_key
        self.model = model
        self.retries = retries
        self.allow_insecure_ssl = allow_insecure_ssl
        # Running token/call totals for cost estimation (see pricing.py).
        self.calls = 0
        self.prompt_tokens = 0
        self.output_tokens = 0

    def _api_url(self, method: str = "generateContent") -> str:
        return (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quote(self.model)}:{method}?key={quote(self.api_key)}"
        )

    def _track_usage(self, data: dict) -> None:
        """Accumulate token usage reported by the Gemini response so the run can
        report an accurate (not just modelled) cost."""
        self.calls += 1
        usage = (data or {}).get("usageMetadata") or {}
        try:
            self.prompt_tokens += int(usage.get("promptTokenCount", 0) or 0)
            # candidatesTokenCount = visible output; thoughtsTokenCount = billed
            # thinking tokens (2.5 models). Both are billed at the output rate.
            self.output_tokens += int(usage.get("candidatesTokenCount", 0) or 0)
            self.output_tokens += int(usage.get("thoughtsTokenCount", 0) or 0)
        except (TypeError, ValueError):
            pass

    def extract(self, target_name: str, context: str, results: list[SearchResult]) -> list[RawCandidate]:
        # Drop off-topic pages before spending a call — they only add noise.
        results = _relevant_results(target_name, results)
        if not results:
            return []
        payload = {
            "system_instruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": _evidence_prompt(target_name, context, results)}]}],
            "generationConfig": {"temperature": 0.1, "response_mime_type": "application/json"},
        }
        data = request_json(
            self._api_url(),
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            timeout=90,
            retries=self.retries,
            allow_insecure_ssl=self.allow_insecure_ssl,
            label="gemini",
        )
        self._track_usage(data)
        try:
            parts = data["candidates"][0]["content"].get("parts", [])
        except (KeyError, IndexError):
            return []
        text = "\n".join(part.get("text", "") for part in parts)
        parsed = _parse_json_object(text)

        evidence_urls = {r.url for r in results}
        target_key = normalize_name(target_name)
        out: list[RawCandidate] = []
        for person in parsed.get("people", []) or []:
            name = squeeze(str(person.get("name", "")))
            citation = str(person.get("citation_url", "")).strip()
            if not name or normalize_name(name) == target_key:
                continue
            if citation not in evidence_urls:
                continue
            if not _plausible_person_name(name):
                continue  # free backstop: drop headlines/labels the model slipped in
            explanation = squeeze(str(person.get("relationship", "")))
            ev_quote = squeeze(str(person.get("evidence_quote", "")))
            if is_sensitive(explanation, ev_quote, name):
                continue
            try:
                confidence = float(person.get("confidence", 0) or 0)
            except (TypeError, ValueError):
                confidence = 0.0
            out.append(RawCandidate(
                name=name,
                explanation=explanation or f"Publicly connected to {target_name}.",
                signal_category=_clean_category(person.get("signal_category", ""), "incidental"),
                confidence=max(0.0, min(1.0, confidence)),
                citation_url=citation,
                evidence_quote=ev_quote[:280],
                published_date=to_iso_date(str(person.get("published_date", "")).strip()),
                method="gemini",
                prominence=str(person.get("prominence", "")).strip().lower(),
            ))
        return out

    def verify_candidates(
        self,
        target_name: str,
        context: str,
        candidates: list[RawCandidate],
    ) -> list[RawCandidate]:
        """AI verification pass: keep only real people with a genuine connection.

        Filters out phrases, company names, navigation text, and any other
        non-person strings that extraction may have emitted. One lightweight
        Gemini call per batch.
        """
        if not candidates:
            return []
        lines = []
        for i, c in enumerate(candidates, 1):
            evidence = c.evidence_quote or c.explanation
            lines.append(f'{i}. Name: "{c.name}"\n   Evidence: {evidence[:200]}')
        prompt = (
            f"TARGET: {target_name}\nCONTEXT: {context}\n\n"
            "Verify which of these are real people with a genuine connection to TARGET:\n\n"
            + "\n\n".join(lines)
        )
        payload = {
            "system_instruction": {"parts": [{"text": _VERIFY_SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "response_mime_type": "application/json"},
        }
        try:
            data = request_json(
                self._api_url(),
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                timeout=60,
                retries=self.retries,
                allow_insecure_ssl=self.allow_insecure_ssl,
                label="gemini-verify",
            )
        except HttpError:
            return candidates  # if verify call fails, pass all through
        self._track_usage(data)
        try:
            parts = data["candidates"][0]["content"].get("parts", [])
        except (KeyError, IndexError):
            return candidates
        parsed = _parse_json_object("\n".join(p.get("text", "") for p in parts))
        raw_valid = parsed.get("valid_names")
        # Fail open unless we got a real list — a scalar string would otherwise
        # iterate into single characters and silently reject every candidate.
        if not isinstance(raw_valid, list):
            return candidates
        valid = {normalize_name(n) for n in raw_valid if isinstance(n, str) and n}
        return [c for c in candidates if normalize_name(c.name) in valid]

    def extract_from_image(self, target_name: str, context: str, image: ImageRef,
                           *, page_date: str = "", min_confidence: float = 0.5) -> list[RawCandidate]:
        """Vision-analyze one uncaptioned image; return named people it contains."""
        downloaded = download_image(image.url, allow_insecure_ssl=self.allow_insecure_ssl)
        if not downloaded:
            return []
        raw_bytes, mime = downloaded

        prompt = (
            f"TARGET: {target_name}\nCONTEXT (disambiguation): {context}\n"
            f"Source page: {image.page_url}\n\n"
            "Identify named people in this photo per the rules. They are potential "
            "connections to the TARGET by appearing in the same photo."
        )
        payload = {
            "system_instruction": {"parts": [{"text": _VISION_SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime, "data": base64.b64encode(raw_bytes).decode("ascii")}},
            ]}],
            "generationConfig": {"temperature": 0.0, "response_mime_type": "application/json"},
        }
        try:
            data = request_json(
                self._api_url(),
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload).encode("utf-8"), method="POST",
                timeout=120, retries=self.retries,
                allow_insecure_ssl=self.allow_insecure_ssl, label="gemini-vision",
            )
        except HttpError:
            return []
        self._track_usage(data)
        try:
            parts = data["candidates"][0]["content"].get("parts", [])
        except (KeyError, IndexError):
            return []
        parsed = _parse_json_object("\n".join(p.get("text", "") for p in parts))

        target_key = normalize_name(target_name)
        out: list[RawCandidate] = []
        for person in parsed.get("people", []) or []:
            name = squeeze(str(person.get("name", "")))
            basis = str(person.get("basis", "")).strip().lower()
            if not name or normalize_name(name) == target_key:
                continue
            if basis not in ("visible_text", "recognized_public_figure"):
                continue
            # Require explicit adult confirmation — no minors.
            if not person.get("appears_adult"):
                continue
            seen = squeeze(str(person.get("what_you_see", "")))
            if is_sensitive(seen, name):
                continue
            try:
                confidence = float(person.get("confidence", 0) or 0)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < min_confidence:
                continue
            basis_label = "name visible in photo" if basis == "visible_text" else "recognized public figure"
            out.append(RawCandidate(
                name=name,
                explanation=f"Appears in an uncaptioned photo with {target_name} ({basis_label}).",
                signal_category="joint_appearance",
                confidence=max(0.0, min(1.0, confidence)) * 0.9,
                citation_url=image.page_url or image.url,
                evidence_quote=f"Photo evidence ({basis_label}): {seen} [{image.url}]"[:280],
                published_date=to_iso_date(page_date),
                method="gemini_vision",
            ))
        return out


def photo_candidates(
    target_name: str,
    context: str,
    images: list[ImageRef],
    *,
    gemini: GeminiExtractor | None,
    max_photos: int,
    page_date: str = "",
    verbose: bool = True,
) -> list[RawCandidate]:
    """Analyze up to ``max_photos`` UNCAPTIONED images and return named people."""
    if gemini is None or max_photos <= 0 or not images:
        return []
    out: list[RawCandidate] = []
    analyzed = 0
    for image in images:
        if analyzed >= max_photos:
            break
        if image.has_caption():
            continue
        analyzed += 1
        if verbose:
            print(f"    [vision] analyzing photo: {image.url}", file=sys.stderr)
        try:
            out.extend(gemini.extract_from_image(target_name, context, image, page_date=page_date))
        except Exception as error:
            if verbose:
                print(f"    [vision] failed for {image.url}: {error}", file=sys.stderr)
    return out


def _chunk(items: list, size: int):
    for i in range(0, len(items), max(1, size)):
        yield items[i:i + size]


def _batch_cache_key(target_name: str, context: str, model: str,
                     batch: list[SearchResult]) -> str:
    """Stable key for one extraction batch so identical evidence never pays for a
    second Gemini call (across re-runs). Keyed on the URLs, not page text, since
    the search cache already pins URLs for a given query set."""
    urls = "|".join(sorted({r.url for r in batch}))
    raw = "|".join([normalize_name(target_name), context.strip().lower(), model, urls])
    return "extract:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def extract_candidates(
    target_name: str,
    context: str,
    results: list[SearchResult],
    *,
    gemini: GeminiExtractor | None,
    batch_size: int = 8,
    deep_verify: bool = False,
    cache=None,
    verbose: bool = True,
) -> list[RawCandidate]:
    """Extract connected people from ALL evidence in a few batched Gemini calls.

    Cost controls:
    - Evidence is deduped/relevance-filtered upstream and processed in batches of
      ``batch_size`` so one run makes a handful of calls, not one per query.
    - ``cache`` (a Store) memoizes each batch by its evidence URLs, so re-runs
      that hit the search cache pay nothing for extraction.
    - The separate LLM verification pass is OFF by default (``deep_verify``); the
      strict extraction prompt plus a free person-name backstop already filter
      non-people. Turn it on for a paranoid, higher-cost pass.

    Falls back to the no-LLM heuristic per batch on a Gemini error.
    """
    if not results:
        return []
    if gemini is None:
        return heuristic_extract(target_name, results)

    relevant = _relevant_results(target_name, results)
    if not relevant:
        return []

    out: list[RawCandidate] = []
    for batch in _chunk(relevant, batch_size):
        key = _batch_cache_key(target_name, context, gemini.model, batch) if cache else None
        if key is not None:
            cached = cache.get_extraction(key)
            if cached is not None:
                out.extend(cached)
                continue
        used_fallback = False
        try:
            extracted = gemini.extract(target_name, context, batch)
        except HttpError as error:
            if verbose:
                print(f"  [gemini] extraction failed ({error}); heuristic fallback for this batch.",
                      file=sys.stderr)
            extracted = heuristic_extract(target_name, batch)
            used_fallback = True
        # Only cache genuine Gemini output, never a transient-failure fallback.
        if key is not None and not used_fallback:
            cache.put_extraction(key, extracted)
        out.extend(extracted)

    if deep_verify and out and gemini is not None:
        try:
            out = gemini.verify_candidates(target_name, context, out)
        except HttpError:
            pass
    return out
