#!/usr/bin/env python3
"""Offline filter for scoring .com auction domains.

This script implements the MVP behaviour described in the project
specification.  It loads an `auction.csv` file that contains a `name`
column, normalises the domains, scores them according to a mixture of
dictionary checks and heuristic bonuses, and writes two CSV files:
`domler_filtered.csv` with every surviving row and
`domler_calllist.csv` with the top 100.

The implementation favours clarity and traceability over micro
optimisations – the dataset is processed row-by-row with tqdm progress
feedback so the behaviour is easy to follow locally.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Dict, List, Optional, Set

import pandas as pd
from tqdm import tqdm


AUCTION_FILE = "auction.csv"
DICTIONARY_FILE = "dictionary.txt"
OUTPUT_FILTERED = "domler_filtered.csv"
OUTPUT_CALLLIST = "domler_calllist.csv"

TRADEMARK_BLACKLIST = {
    "google",
    "apple",
    "openai",
    "microsoft",
    "amazon",
    "facebook",
    "meta",
    "tesla",
    "uber",
}

SUFFIX_BONUS_SUFFIXES = ("ly", "ify", "able", "ster", "ing")

TECH_TERMS = {
    "ai",
    "data",
    "cloud",
    "lab",
    "labs",
    "robot",
    "robo",
    "robotics",
    "meta",
    "gen",
    "bio",
    "ops",
    "dev",
    "vr",
    "xr",
    "ml",
    "neuro",
    "quantum",
    "nova",
}

NATURE_TERMS = {
    "oak",
    "pine",
    "cedar",
    "maple",
    "river",
    "brook",
    "stone",
    "rock",
    "forest",
    "flora",
    "fauna",
    "leaf",
    "green",
    "blue",
    "red",
    "gold",
    "silver",
    "wolf",
    "fox",
    "bear",
    "lynx",
    "owl",
    "hawk",
    "falcon",
    "eagle",
    "lion",
    "tiger",
    "rain",
    "snow",
    "sun",
    "solar",
    "wind",
}

VALUES_TERMS = {
    "trust",
    "true",
    "honest",
    "calm",
    "swift",
    "rapid",
    "bolt",
    "nova",
    "noble",
    "prime",
    "fair",
    "clear",
    "bright",
    "vivid",
    "brave",
    "kind",
    "bold",
    "luxe",
    "royal",
    "alpha",
    "neo",
    "next",
    "future",
}

COMPOUND_MINI_WORDS = {
    "roof",
    "shop",
    "clean",
    "data",
    "cloud",
    "lab",
    "labs",
    "bank",
    "pay",
    "cash",
    "box",
    "desk",
    "care",
    "home",
    "tech",
    "bio",
    "med",
    "card",
    "mark",
    "ship",
    "pack",
    "wave",
    "light",
    "safe",
    "guard",
    "leaf",
    "stone",
}

VOWELS = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")


def load_dictionary(path: str) -> Set[str]:
    if not os.path.exists(path):
        print(f"Error: required dictionary file '{path}' was not found.", file=sys.stderr)
        sys.exit(1)

    words: Set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            word = line.strip().lower()
            if 3 <= len(word) <= 10 and word.isalpha():
                words.add(word)

    if not words:
        print(f"Error: dictionary '{path}' is empty or contains no valid words.", file=sys.stderr)
        sys.exit(1)

    return words


def extract_sld(domain: str) -> Optional[str]:
    domain = domain.strip().lower()
    if not domain or not domain.endswith(".com"):
        return None
    sld = domain.split(".", 1)[0]
    if not sld.isalpha():
        return None
    if len(sld) < 3 or len(sld) > 15:
        return None
    if any(mark in sld for mark in TRADEMARK_BLACKLIST):
        return None
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{4,}", sld):
        return None
    if re.search(r"(.)\1\1", sld):
        return None

    has_vowel = any(ch in VOWELS for ch in sld)
    has_y_vowel = bool(re.search(r"[bcdfghjklmnpqrstvwxyz]y[bcdfghjklmnpqrstvwxyz]", sld))
    if not has_vowel and not has_y_vowel:
        return None

    return sld


def edge_dictionary_hit(sld: str, dictionary_words: Set[str]) -> bool:
    for length in range(3, min(10, len(sld)) + 1):
        prefix = sld[:length]
        suffix = sld[-length:]
        if prefix in dictionary_words or suffix in dictionary_words:
            return True
    return False


def full_segmentation(sld: str, dictionary_words: Set[str]) -> bool:
    n = len(sld)
    if n < 3:
        return False

    dp = [False] * (n + 1)
    dp[0] = True

    for i in range(3, n + 1):
        for length in range(3, min(10, i) + 1):
            j = i - length
            if j >= 0 and dp[j] and sld[j:i] in dictionary_words:
                dp[i] = True
                break

    return dp[n]


def single_word_bonus(length: int) -> int:
    if 4 <= length <= 8:
        return 3
    if 9 <= length <= 10:
        return 2
    if 11 <= length <= 12:
        return 1
    return 0


def suffix_bonus(sld: str) -> int:
    for suffix in SUFFIX_BONUS_SUFFIXES:
        if sld.endswith(suffix) and len(sld) - len(suffix) >= 3:
            return 2
    return 0


def substring_bonus(sld: str, terms: Set[str]) -> int:
    for term in terms:
        if term in sld:
            return 1
    return 0


def compound_bonus(sld: str) -> int:
    length = len(sld)
    for split in range(3, length - 2):
        left = sld[:split]
        right = sld[split:]
        if left in COMPOUND_MINI_WORDS and right in COMPOUND_MINI_WORDS:
            return 2
    return 0


def misspelling_bonus(sld: str) -> int:
    has_vowel = any(ch in VOWELS for ch in sld)
    y_pattern = re.search(r"[bcdfghjklmnpqrstvwxyz]y[bcdfghjklmnpqrstvwxyz]", sld)
    if not has_vowel:
        if y_pattern:
            return 1
        return -1
    return 0


def calculate_score_components(sld: str, dictionary_words: Set[str]) -> Dict[str, int | bool]:
    length = len(sld)
    dict_edge = edge_dictionary_hit(sld, dictionary_words)
    full_seg = full_segmentation(sld, dictionary_words)

    dict_bonus_value = 0
    if dict_edge or full_seg:
        dict_bonus_value = 8 if full_seg else 5

    components = {
        "dict_hits_edge": dict_edge,
        "full_segmentation": full_seg,
        "dict_bonus": dict_bonus_value,
        "single_bonus": single_word_bonus(length),
        "suffix_bonus": suffix_bonus(sld),
        "tech_bonus": substring_bonus(sld, TECH_TERMS),
        "nature_bonus": substring_bonus(sld, NATURE_TERMS),
        "values_bonus": substring_bonus(sld, VALUES_TERMS),
        "compound_bonus": compound_bonus(sld),
        "misspelling": misspelling_bonus(sld),
        "len_penalty": -1 if length > 12 else 0,
    }

    total = (
        components["dict_bonus"]
        + components["single_bonus"]
        + components["suffix_bonus"]
        + components["tech_bonus"]
        + components["nature_bonus"]
        + components["values_bonus"]
        + components["compound_bonus"]
        + components["misspelling"]
        + components["len_penalty"]
    )

    components["score"] = total
    components["len"] = length
    return components


def ensure_required_columns(df: pd.DataFrame) -> None:
    if "name" not in df.columns:
        print("Error: auction.csv must contain a 'name' column.", file=sys.stderr)
        sys.exit(1)


def process_auction(dictionary_words: Set[str]) -> pd.DataFrame:
    df = pd.read_csv(AUCTION_FILE)
    ensure_required_columns(df)

    optional_columns = [col for col in ["price_usd", "bid_count", "end_date", "permalink"] if col in df.columns]

    rows: List[Dict[str, object]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring domains"):
        domain_raw = str(row["name"]).strip()
        sld = extract_sld(domain_raw)
        if not sld:
            continue

        components = calculate_score_components(sld, dictionary_words)

        if not (components["dict_hits_edge"] or components["full_segmentation"]):
            continue

        if components["score"] < 8:
            continue

        record: Dict[str, object] = {
            "domain": domain_raw.lower(),
            "sld": sld,
            "len": components["len"],
            "score": components["score"],
            "dict_hits_edge": components["dict_hits_edge"],
            "full_segmentation": components["full_segmentation"],
            "dict_bonus": components["dict_bonus"],
            "single_bonus": components["single_bonus"],
            "suffix_bonus": components["suffix_bonus"],
            "tech_bonus": components["tech_bonus"],
            "nature_bonus": components["nature_bonus"],
            "values_bonus": components["values_bonus"],
            "compound_bonus": components["compound_bonus"],
            "misspelling": components["misspelling"],
            "len_penalty": components["len_penalty"],
        }

        for col in optional_columns:
            record[col] = row.get(col)

        rows.append(record)

    if not rows:
        return pd.DataFrame(
            columns=[
                "domain",
                "sld",
                "len",
                "score",
                "dict_hits_edge",
                "full_segmentation",
                "dict_bonus",
                "single_bonus",
                "suffix_bonus",
                "tech_bonus",
                "nature_bonus",
                "values_bonus",
                "compound_bonus",
                "misspelling",
                "len_penalty",
            ]
            + optional_columns
        )

    filtered_df = pd.DataFrame(rows)
    sort_columns = [
        ("score", False),
        ("dict_bonus", False),
        ("single_bonus", False),
        ("suffix_bonus", False),
        ("len", True),
    ]

    filtered_df = filtered_df.sort_values(
        by=[col for col, _ in sort_columns],
        ascending=[asc for _, asc in sort_columns],
        kind="mergesort",
    )

    return filtered_df


def main() -> None:
    dictionary_words = load_dictionary(DICTIONARY_FILE)

    if not os.path.exists(AUCTION_FILE):
        print(f"Error: required auction file '{AUCTION_FILE}' was not found.", file=sys.stderr)
        sys.exit(1)

    filtered = process_auction(dictionary_words)

    filtered.to_csv(OUTPUT_FILTERED, index=False)

    calllist = filtered.head(100)
    calllist.to_csv(OUTPUT_CALLLIST, index=False)

    print(
        f"Wrote {len(filtered)} rows to {OUTPUT_FILTERED} and {len(calllist)} rows to {OUTPUT_CALLLIST}."
    )


if __name__ == "__main__":
    main()

