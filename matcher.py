#!/usr/bin/env python3
"""Matcher CLI that uses Google Places with a simple JSON cache.

The script consumes the filtered domain list produced by ``domler.py``
and attempts to find potential business matches using the Places Text
Search and Place Details APIs.  Query and details responses are cached
locally so subsequent runs only issue API calls for unknown entries.

The behaviour matches the MVP specification provided in the project
overview, including API call budgeting, cache management flags, scoring
rules, and CSV output layout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable: Iterable, **_kwargs):  # type: ignore
        return iterable

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - fallback path for environments without rapidfuzz
    from difflib import SequenceMatcher

    def _ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio() * 100

    class _FuzzModule:
        @staticmethod
        def ratio(a: str, b: str) -> float:
            return _ratio(a, b)

    fuzz = _FuzzModule()


class BudgetExceeded(RuntimeError):
    """Raised when the API call budget is exhausted."""

    def __init__(self, message: str, results: List[Dict[str, object]]):
        super().__init__(message)
        self.results = results


TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

DEFAULT_LIMIT = 400
DEFAULT_MIN_SCORE = 4
DEFAULT_PER_DOMAIN = 10

OUTPUT_COLUMNS = [
    "total_score",
    "our_domain",
    "website",
    "business_name",
    "phone",
    "country_code",
    "end_date",
    "address",
    "rating",
    "reviews",
    "name_pts",
    "site_pts",
    "tld_bonus",
    "hyphen_bonus",
    "edge_bonus",
    "sld",
    "query_used",
    "place_id",
    "source",
]


class ApiRequestError(RuntimeError):
    """Wrapper exception for HTTP errors."""


def load_api_key(path: str) -> str:
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if api_key:
        return api_key.strip()

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            line = handle.readline().strip()
            if line:
                return line

    raise SystemExit(
        "Error: Google API key not provided. Set GOOGLE_MAPS_API_KEY or create 'api.txt' with the key on the first line."
    )


def ensure_cache_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return {}


def write_json(path: str, data: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def host_only(url: str) -> str:
    url = url.strip()
    if not url:
        return ""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = (parsed.netloc or parsed.path).lower()
    if host.startswith("www."):
        host = host[4:]
    return host.rstrip(".")


def root_domain(host: str) -> str:
    parts = [p for p in host.split(".") if p]
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def normalized_output_website(url: str) -> str:
    host = host_only(url)
    return root_domain(host)


def website_sld(website: str) -> str:
    parts = website.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return website


def normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def similarity_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 20.0


def extract_country_code(detail: Dict[str, object]) -> str:
    """Return a two-letter country code if one can be inferred."""

    components = detail.get("address_components")
    if isinstance(components, list):
        for component in components:
            if not isinstance(component, dict):
                continue
            types = component.get("types", [])
            if not isinstance(types, list) or "country" not in types:
                continue
            code = component.get("short_name")
            if isinstance(code, str) and code.strip():
                return code.strip().upper()

    return ""


def compute_candidate_score(sld: str, business_name: str, website: str) -> Tuple[float, Dict[str, float]]:
    norm_name = normalise_name(business_name)
    name_pts = similarity_score(sld, norm_name)

    website_root = normalized_output_website(website)
    site_sld = website_sld(website_root)
    if site_sld == sld and sld:
        breakdown = {
            "name_pts": 0.0,
            "site_pts": 0.0,
            "tld_bonus": 0.0,
            "hyphen_bonus": 0.0,
            "edge_bonus": 0.0,
            "total_score": 100.0,
        }
        return 100.0, breakdown

    site_pts = similarity_score(sld, site_sld)

    tld = website_root.split(".")[-1] if website_root else ""
    tld_bonus = 2.0 if tld and tld != "com" else 0.0
    hyphen_bonus = 1.0 if "-" in website_root else 0.0

    edge_bonus = 0.0
    if site_sld.startswith(sld) or site_sld.endswith(sld):
        edge_bonus += 2.0
    if norm_name.startswith(sld) or norm_name.endswith(sld):
        edge_bonus += 1.0

    total = min(100.0, name_pts + site_pts + tld_bonus + hyphen_bonus + edge_bonus)

    breakdown = {
        "name_pts": round(name_pts, 3),
        "site_pts": round(site_pts, 3),
        "tld_bonus": tld_bonus,
        "hyphen_bonus": hyphen_bonus,
        "edge_bonus": edge_bonus,
        "total_score": round(total, 3),
    }
    return total, breakdown


def _http_get_json(url: str, params: Dict[str, str]) -> Dict[str, object]:
    query = urllib.parse.urlencode(params)
    target = f"{url}?{query}"
    try:
        with urllib.request.urlopen(target, timeout=30) as handle:
            payload = handle.read()
    except urllib.error.URLError as exc:  # pragma: no cover - network failure path
        raise ApiRequestError(str(exc)) from exc

    try:
        return json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - invalid JSON path
        raise ApiRequestError(f"Invalid JSON response: {exc}") from exc


def text_search(query: str, api_key: str, region: Optional[str]) -> Dict[str, object]:
    params = {"query": query, "key": api_key}
    if region:
        params["region"] = region
    return _http_get_json(TEXT_SEARCH_URL, params)


def place_details(place_id: str, api_key: str) -> Dict[str, object]:
    params = {
        "place_id": place_id,
        "key": api_key,
        "fields": (
            "name,formatted_phone_number,formatted_address,website,rating," "user_ratings_total,address_components"
        ),
    }
    return _http_get_json(DETAILS_URL, params)


def maybe_increment(api_calls: int, limit: Optional[int]) -> int:
    if limit is not None and api_calls >= limit:
        raise RuntimeError("API call budget exhausted")
    return api_calls + 1


@dataclass
class CacheResult:
    place_ids: Sequence[str]
    source: str  # "api" or "cache"


def fetch_place_ids(
    sld: str,
    region: Optional[str],
    api_key: str,
    queries_cache: Dict[str, object],
    refresh: bool,
    api_calls: int,
    limit: Optional[int],
) -> Tuple[Optional[CacheResult], Dict[str, object], int]:
    key = f"{sld}|{region or ''}"
    if not refresh and key in queries_cache:
        entry = queries_cache[key]
        if entry.get("status") == "no_results":
            return None, queries_cache, api_calls
        return CacheResult(entry.get("place_ids", []), "cache"), queries_cache, api_calls

    api_calls = maybe_increment(api_calls, limit)
    try:
        response = text_search(sld, api_key, region)
    except ApiRequestError as exc:  # pragma: no cover - network failure path
        print(f"Warning: text search failed for {sld}: {exc}", file=sys.stderr)
        queries_cache[key] = {"status": "no_results", "ts": time.time()}
        return None, queries_cache, api_calls

    status = response.get("status", "")
    results = response.get("results", []) if status == "OK" else []
    place_ids = [item.get("place_id") for item in results if item.get("place_id")]

    if place_ids:
        queries_cache[key] = {"status": "ok", "place_ids": place_ids, "ts": time.time()}
        return CacheResult(place_ids, "api"), queries_cache, api_calls

    queries_cache[key] = {"status": "no_results", "ts": time.time()}
    return None, queries_cache, api_calls


def fetch_place_details(
    place_id: str,
    api_key: str,
    details_cache: Dict[str, object],
    refresh: bool,
    api_calls: int,
    limit: Optional[int],
) -> Tuple[Optional[Dict[str, object]], Dict[str, object], int, str]:
    if not refresh and place_id in details_cache:
        cached = details_cache[place_id]
        return cached.get("result"), details_cache, api_calls, "cache"

    api_calls = maybe_increment(api_calls, limit)
    try:
        response = place_details(place_id, api_key)
    except ApiRequestError as exc:  # pragma: no cover - network failure path
        print(f"Warning: details request failed for {place_id}: {exc}", file=sys.stderr)
        return None, details_cache, api_calls, "api"

    if response.get("status") != "OK":
        return None, details_cache, api_calls, "api"

    result = response.get("result")
    if result is not None:
        details_cache[place_id] = {"result": result, "ts": time.time()}
    return result, details_cache, api_calls, "api"


def load_filtered_domains(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise SystemExit(f"Error: filtered domains file '{path}' not found.")

    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit("Error: filtered domains file has no header row.")

        required = {"domain", "sld"}
        missing = required - set(reader.fieldnames)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise SystemExit(f"Error: filtered domains file missing required columns: {missing_list}")

        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append(row)

    return rows


def process_domain(
    domain: str,
    sld: str,
    end_date: str,
    region: Optional[str],
    api_key: str,
    queries_cache: Dict[str, object],
    details_cache: Dict[str, object],
    refresh_queries: bool,
    refresh_details: bool,
    limit: Optional[int],
    api_calls: int,
    min_match_score: float,
    per_domain: int,
    used_websites: set,
) -> Tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object], int]:
    cache_result, queries_cache, api_calls = fetch_place_ids(
        sld, region, api_key, queries_cache, refresh_queries, api_calls, limit
    )

    if cache_result is None:
        return [], queries_cache, details_cache, api_calls

    place_ids = list(cache_result.place_ids)[:20]
    domain_results: List[Dict[str, object]] = []

    for place_id in place_ids:
        detail, details_cache, api_calls, detail_source = fetch_place_details(
            place_id, api_key, details_cache, refresh_details, api_calls, limit
        )

        if not detail:
            continue

        website = detail.get("website")
        if not website:
            continue

        normalized_website = normalized_output_website(website)
        if not normalized_website or normalized_website in used_websites:
            continue

        total, breakdown = compute_candidate_score(sld, detail.get("name", ""), normalized_website)
        if total < min_match_score:
            continue

        phone = detail.get("formatted_phone_number") or detail.get("international_phone_number")

        candidate = {
            "total_score": breakdown["total_score"],
            "our_domain": domain,
            "website": normalized_website,
            "business_name": detail.get("name"),
            "phone": phone,
            "country_code": extract_country_code(detail),
            "end_date": end_date,
            "address": detail.get("formatted_address"),
            "rating": detail.get("rating"),
            "reviews": detail.get("user_ratings_total"),
            "name_pts": breakdown["name_pts"],
            "site_pts": breakdown["site_pts"],
            "tld_bonus": breakdown["tld_bonus"],
            "hyphen_bonus": breakdown["hyphen_bonus"],
            "edge_bonus": breakdown["edge_bonus"],
            "sld": sld,
            "query_used": f"{sld}|{region or ''}",
            "place_id": place_id,
            "source": "cache" if (cache_result.source == "cache" and detail_source == "cache") else "api",
        }

        domain_results.append(candidate)

    domain_results.sort(key=lambda item: item["total_score"], reverse=True)

    selected: List[Dict[str, object]] = []
    limit_per_domain = float("inf") if per_domain <= 0 else per_domain
    for candidate in domain_results:
        if candidate["website"] in used_websites:
            continue
        selected.append(candidate)
        used_websites.add(candidate["website"])
        if len(selected) >= limit_per_domain:
            break

    return selected, queries_cache, details_cache, api_calls


def gather_matches(
    rows: Sequence[Dict[str, object]],
    region: Optional[str],
    api_key: str,
    cache_dir: str,
    refresh_queries: bool,
    refresh_details: bool,
    limit: Optional[int],
    min_match_score: float,
    per_domain: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object]]:
    ensure_cache_dir(cache_dir)
    queries_path = os.path.join(cache_dir, "queries.json")
    details_path = os.path.join(cache_dir, "details.json")

    queries_cache = read_json(queries_path)
    details_cache = read_json(details_path)

    used_websites: set = set()
    api_calls = 0
    all_results: List[Dict[str, object]] = []

    try:
        total = len(rows)
        for row in tqdm(rows, total=total, desc="Matching domains"):
            domain = str(row.get("domain", "")).strip()
            sld = str(row.get("sld", "")).strip().lower()
            end_date = str(row.get("end_date", "")).strip()
            if not domain or not sld:
                continue

            results, queries_cache, details_cache, api_calls = process_domain(
                domain,
                sld,
                end_date,
                region,
                api_key,
                queries_cache,
                details_cache,
                refresh_queries,
                refresh_details,
                limit,
                api_calls,
                min_match_score,
                per_domain,
                used_websites,
            )

            all_results.extend(results)
    except RuntimeError as exc:
        write_json(queries_path, queries_cache)
        write_json(details_path, details_cache)
        raise BudgetExceeded(str(exc), all_results) from exc

    write_json(queries_path, queries_cache)
    write_json(details_path, details_cache)

    return all_results, queries_cache, details_cache


def format_stats(queries_cache: Dict[str, object], details_cache: Dict[str, object]) -> str:
    queries_total = len(queries_cache)
    no_results = sum(1 for entry in queries_cache.values() if entry.get("status") == "no_results")
    details_total = len(details_cache)
    return (
        f"Cache stats -> queries: {queries_total} total, {no_results} no_results; "
        f"details: {details_total} entries"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match filtered domains with potential businesses via Google Places.")
    parser.add_argument("--input", default="domler_filtered.csv", help="Input CSV from domler.py")
    parser.add_argument("--output", default="domler_matches.csv", help="Output CSV path")
    parser.add_argument("--region", default=None, help="Optional region code for Text Search (e.g. us)")
    parser.add_argument("--min_match_score", type=float, default=DEFAULT_MIN_SCORE, help="Minimum total score to keep a match")
    parser.add_argument("--per_domain", type=int, default=DEFAULT_PER_DOMAIN, help="Maximum matches per domain")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="API call limit (Text Search + Details)")
    parser.add_argument("--cache-dir", default=".domler_cache_cq", help="Cache directory")
    parser.add_argument("--refresh-queries", action="store_true", help="Ignore cached query responses")
    parser.add_argument("--refresh-details", action="store_true", help="Ignore cached place details")
    parser.add_argument("--nuke-cache", action="store_true", help="Delete the cache directory before running")
    parser.add_argument("--stats", action="store_true", help="Print cache statistics after the run")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if args.nuke_cache and os.path.exists(args.cache_dir):
        shutil.rmtree(args.cache_dir)

    api_key = load_api_key("api.txt")
    df = load_filtered_domains(args.input)

    limit = None if args.limit is not None and args.limit < 0 else args.limit

    try:
        results, queries_cache, details_cache = gather_matches(
            df,
            args.region,
            api_key,
            args.cache_dir,
            args.refresh_queries,
            args.refresh_details,
            limit,
            args.min_match_score,
            args.per_domain,
        )
    except BudgetExceeded as exc:
        print(f"Stopped early: {exc}", file=sys.stderr)
        results = exc.results
        queries_cache = read_json(os.path.join(args.cache_dir, "queries.json"))
        details_cache = read_json(os.path.join(args.cache_dir, "details.json"))
    except RuntimeError as exc:
        print(f"Stopped early due to error: {exc}", file=sys.stderr)
        results = []
        queries_cache = read_json(os.path.join(args.cache_dir, "queries.json"))
        details_cache = read_json(os.path.join(args.cache_dir, "details.json"))

    with open(args.output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Wrote {len(results)} matches to {args.output}.")

    if args.stats:
        print(format_stats(queries_cache, details_cache))


if __name__ == "__main__":
    main()

