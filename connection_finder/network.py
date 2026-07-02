from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .util import name_tokens, normalize_name, squeeze


def _first_value(row: dict, keys: list[str]) -> str:
    normalized = {(k or "").lower().strip(): v for k, v in row.items()}
    for key in keys:
        value = normalized.get(key.lower())
        if value:
            return squeeze(value)
    return ""


def _first_last_key(name: str) -> str:
    tokens = name_tokens(name)
    if len(tokens) < 2:
        return ""
    return f"{tokens[0]} {tokens[-1]}"


@dataclass
class NetworkMatch:
    in_network: bool = False
    degree: Optional[int] = None
    via: str = ""
    matched_name: str = ""
    approximate: bool = False  # matched on first+last only (middle names differ)


@dataclass
class NetworkIndex:
    first_degree: dict[str, dict] = field(default_factory=dict)
    second_degree: dict[str, list[str]] = field(default_factory=dict)  # name_key -> via names
    _fl_first: dict[str, str] = field(default_factory=dict)   # first+last -> name_key (1st degree)
    _fl_second: dict[str, str] = field(default_factory=dict)  # first+last -> name_key (2nd degree)

    @property
    def size(self) -> tuple[int, int]:
        return len(self.first_degree), len(self.second_degree)

    def lookup(self, name: str) -> NetworkMatch:
        key = normalize_name(name)
        if not key:
            return NetworkMatch()
        if key in self.first_degree:
            return NetworkMatch(True, 1, "", self.first_degree[key].get("name", name))
        if key in self.second_degree:
            via = self.second_degree[key]
            return NetworkMatch(True, 2, ", ".join(via[:3]), name)
        # First+last fallback catches "Robert A. Smith" vs "Robert Smith", but a
        # different person who shares a first+last name would collide — so mark
        # these approximate (hedged label, reduced boost) rather than asserting it.
        fl = _first_last_key(name)
        if fl and fl in self._fl_first:
            mk = self._fl_first[fl]
            return NetworkMatch(True, 1, "", self.first_degree[mk].get("name", name), approximate=True)
        if fl and fl in self._fl_second:
            mk = self._fl_second[fl]
            return NetworkMatch(True, 2, ", ".join(self.second_degree[mk][:3]), name, approximate=True)
        return NetworkMatch()


def load_connections_csv(path: str) -> dict[str, dict]:
    """Load a LinkedIn 'Connections.csv' export into {name_key: profile}.

    Tolerant of the few note lines LinkedIn prepends before the real header.
    """
    people: dict[str, dict] = {}
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(path)
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        start = 0
        for line_no, line in enumerate(sample.splitlines()):
            low = line.lower()
            if "first name" in low or low.startswith("name,") or low == "name":
                start = line_no
                break
        for _ in range(start):
            next(handle, None)
        for row in csv.DictReader(handle):
            name = _first_value(row, ["name", "full name"])
            if not name:
                first = _first_value(row, ["first name"])
                last = _first_value(row, ["last name"])
                name = " ".join(p for p in [first, last] if p).strip()
            key = normalize_name(name)
            if not key or key in people:
                continue
            people[key] = {
                "name": name,
                "url": _first_value(row, ["url", "profile url", "linkedin url"]),
                "company": _first_value(row, ["company", "current company"]),
                "title": _first_value(row, ["position", "title", "job title"]),
            }
    return people


def load_second_degree(path: str, first_degree: dict[str, dict]) -> dict[str, list[str]]:
    """Load {your_connection_name: [their_connection_names...]} and invert it to
    {second_degree_name_key: [via first-degree names...]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    inverted: dict[str, list[str]] = {}
    if not isinstance(data, dict):
        return inverted
    for via_name, contacts in data.items():
        if not isinstance(contacts, list):
            continue
        for contact in contacts:
            contact_name = contact if isinstance(contact, str) else (contact or {}).get("name", "")
            key = normalize_name(contact_name)
            if not key or key in first_degree:
                continue  # already a direct connection
            inverted.setdefault(key, [])
            if via_name not in inverted[key]:
                inverted[key].append(via_name)
    return inverted


def build_index(connections_csv: str = "", second_degree_json: str = "", *, verbose: bool = True) -> NetworkIndex:
    index = NetworkIndex()
    if connections_csv:
        try:
            index.first_degree = load_connections_csv(connections_csv)
        except FileNotFoundError:
            if verbose:
                print(f"  [network] connections file not found: {connections_csv}", file=sys.stderr)
        except Exception as error:
            if verbose:
                print(f"  [network] could not parse {connections_csv}: {error}", file=sys.stderr)
    if second_degree_json and index.first_degree is not None:
        try:
            index.second_degree = load_second_degree(second_degree_json, index.first_degree)
        except Exception as error:
            if verbose:
                print(f"  [network] could not parse {second_degree_json}: {error}", file=sys.stderr)

    for key, profile in index.first_degree.items():
        fl = _first_last_key(profile.get("name", ""))
        if fl:
            index._fl_first.setdefault(fl, key)
    for key in index.second_degree:
        fl = _first_last_key(key)  # key is already normalized "first ... last"
        if fl:
            index._fl_second.setdefault(fl, key)
    return index
