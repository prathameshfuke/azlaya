"""Step 1: normalize business_name and business_address across all six source files.

Not executed here (no local run per the project constraints) — read the code back and spot-check
it on the GPU machine before trusting it; see the "NEEDS GPU-MACHINE VERIFICATION" notes below for
the specific spots this file cannot self-check offline (regexes were written against the documented
noise patterns, not against real rows).

Run (from code/business_entity_resolution/):
    python -m src.normalize --split train
    python -m src.normalize --split test
    python -m src.normalize --split train --split test   # both, default

Writes data_processed/{split}_{source1,source2,source3}_normalized.tsv with columns:
    entity_id, business_name, business_address, country,
    name_norm, legal_name_norm, trade_name_norm, has_dba,
    address_norm, pin_code, city_guess,
    name_script, address_script
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import re
from typing import Dict, Tuple

import pandas as pd
from tqdm.auto import tqdm

from src import common

# --------------------------------------------------------------------------- legal-suffix dicts
#
# Country-aware and *not* merged: France's "SA"/"SAS"/"SARL" must never be expanded by the
# US/India dict (and vice versa) or e.g. a French "SA" could collide with an unrelated US token.
# Every pattern is a whole-word regex, so "sas" cannot match inside "sasu" etc.

_US_INDIA_SUFFIXES: Dict[str, str] = {
    r"\bcorpn\b": "corporation",
    r"\bcorp\b": "corporation",
    r"\bltd\b": "limited",
    r"\bpvt\b": "private",
    r"\bpriv\b": "private",
    r"\binc\b": "incorporated",
    r"\bincorp\b": "incorporated",
    r"\bco\b": "company",
    r"\bllc\b": "limited liability company",
    r"\bllp\b": "limited liability partnership",
    r"\bplc\b": "public limited company",
}

_FRANCE_SUFFIXES: Dict[str, str] = {
    r"\bsasu\b": "societe par actions simplifiee unipersonnelle",
    r"\bsas\b": "societe par actions simplifiee",
    r"\bsarl\b": "societe a responsabilite limitee",
    r"\beurl\b": "entreprise unipersonnelle a responsabilite limitee",
    r"\bsci\b": "societe civile immobiliere",
    r"\bsa\b": "societe anonyme",
}

_SUFFIX_GROUPS = {"us_india": _US_INDIA_SUFFIXES, "france": _FRANCE_SUFFIXES}
_COMPILED_SUFFIX_GROUPS = {
    group: [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in mapping.items()]
    for group, mapping in _SUFFIX_GROUPS.items()
}


def country_suffix_group(country: str) -> str:
    """Bucket an open-set country label into a legal-suffix dictionary. Only 'France' (case
    insensitive) gets the France dict; every other value -- including 'US', 'India', and any
    unseen future country -- gets the US/India dict, which is the only generic one the challenge
    documents. This does NOT filter or branch on country being one of a fixed set anywhere else
    in the pipeline; it only decides which suffix expansion table to apply."""
    c = (country or "").strip().lower()
    return "france" if c == "france" else "us_india"


def expand_legal_suffixes(name_lower: str, country: str) -> str:
    group = country_suffix_group(country)
    for pattern, replacement in _COMPILED_SUFFIX_GROUPS[group]:
        name_lower = pattern.sub(replacement, name_lower)
    return name_lower


# --------------------------------------------------------------------------------- DBA detection
#
# "X dba Y" / "X d/b/a Y" / "X doing business as Y" / "X trading as Y" / "X t/a Y".
# NEEDS GPU-MACHINE VERIFICATION: run this against a sample of real business_name values and
# check for false positives (e.g. a legitimate name containing "as" as a normal word right after
# something that looks like a legal-entity boundary) before trusting it at scale.
_DBA_RE = re.compile(
    r"^(?P<legal>.+?)\s+(?:d/?\.?b/?\.?a\.?|doing\s+business\s+as|trading\s+as|t/a)\s+(?P<trade>.+)$",
    re.IGNORECASE,
)


def split_dba(raw_name: str) -> Tuple[str, str, bool]:
    """Return (legal_part, trade_part, has_dba). When no DBA pattern is found, both parts equal
    the input and has_dba is False."""
    name = common.collapse_ws(raw_name or "")
    m = _DBA_RE.match(name)
    if not m:
        return name, name, False
    return m.group("legal").strip(), m.group("trade").strip(), True


# ------------------------------------------------------------------------------- name normalizer

_AMP_RE = re.compile(r"&")


def normalize_name_core(raw: str, country: str) -> str:
    if not raw:
        return ""
    text = raw.lower()
    text = _AMP_RE.sub(" and ", text)
    text = common.strip_punctuation(text)
    text = common.collapse_ws(text)
    text = expand_legal_suffixes(text, country)
    text = common.collapse_ws(text)
    return text


def normalize_name_fields(raw_name: str, country: str) -> Dict[str, object]:
    legal_raw, trade_raw, has_dba = split_dba(raw_name)
    legal_norm = normalize_name_core(legal_raw, country)
    trade_norm = normalize_name_core(trade_raw, country) if has_dba else legal_norm
    # name_norm is the general-purpose field blocking/features use by default: the legal name,
    # since it's what's present for every record (trade name only differs when has_dba).
    return {
        "name_norm": legal_norm,
        "legal_name_norm": legal_norm,
        "trade_name_norm": trade_norm,
        "has_dba": has_dba,
    }


# ---------------------------------------------------------------------------- address normalizer
#
# Address abbreviations are treated as generic (not country-gated): the challenge describes
# Rd/St/Ave-style abbreviations without tying them to a specific country, unlike the legal
# suffixes, which explicitly must not be conflated across countries.
_ADDRESS_ABBR: Dict[str, str] = {
    r"\brd\b": "road", r"\bst\b": "street", r"\bave\b": "avenue", r"\bblvd\b": "boulevard",
    r"\bln\b": "lane", r"\bdr\b": "drive", r"\bapt\b": "apartment", r"\bfl\b": "floor",
    r"\bste\b": "suite", r"\bhwy\b": "highway", r"\bpl\b": "place", r"\bct\b": "court",
    r"\bsq\b": "square", r"\bter\b": "terrace", r"\bpkwy\b": "parkway", r"\bcir\b": "circle",
    r"\bmkt\b": "market", r"\bnr\b": "near", r"\bopp\b": "opposite", r"\bbldg\b": "building",
}
_COMPILED_ADDRESS_ABBR = [(re.compile(p, re.IGNORECASE), r) for p, r in _ADDRESS_ABBR.items()]

# Literal placeholder tokens that show up as real field *content* (not true NaN, which the TSV
# reader already keeps as "" via keep_default_na=False) — strip them as standalone tokens so
# "123 Main St, null, 10001" doesn't leave a stray "null" city_guess candidate downstream.
_NULL_TOKEN_RE = re.compile(r"(?i)(?<![a-z])(null|nan|n/?a|none)(?![a-z])")

_INDIA_PIN_RE = re.compile(r"\b\d{6}\b")
_GENERIC_POSTAL_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")


def extract_postal_code(address: str) -> str:
    """India PIN (6 digits) is checked first since it's the more specific pattern; otherwise a
    5-digit (optionally +4) run covers both US ZIP and French postal codes -- disambiguating
    which of those it is isn't needed here, only capturing the digits for blocking/matching.
    Takes the LAST match in the string, since postal codes conventionally trail an address."""
    matches = _INDIA_PIN_RE.findall(address)
    if matches:
        return matches[-1]
    matches = _GENERIC_POSTAL_RE.findall(address)
    return matches[-1] if matches else ""


_DIGITS_RE = re.compile(r"\d+")


def guess_city(address_norm: str, pin_code: str) -> str:
    """Best-effort heuristic, NOT a gazetteer lookup (none is available/allowed for this
    challenge). Two address shapes are common and disagree about where the city sits relative to
    the PIN code's own comma segment:
      - "<street>, <city>, <state> <pin>" or "<street>, <city>, <pin>" (US/India-style): the city
        is the segment BEFORE the one holding the pin.
      - "<street>, <pin> <city>" (France-style, e.g. "12 Rue de Paris, 75001 Paris"): the city is
        IN THE SAME segment as the pin, and the segment before it is the street.
    Disambiguate with one signal: does the segment before the pin's segment look like a street
    (starts with a number, e.g. a house/building number)? If so, prefer extracting the city out of
    the pin's own segment; otherwise treat that previous segment as the city, as before. Falls
    back to the second-to-last comma segment when no PIN was found. NEEDS GPU-MACHINE
    VERIFICATION on the real data: landmark-style addresses ("near SBI ATM") and municipal-
    numbering addresses (no commas at all) will likely still return "" or an imperfect token,
    which is acceptable -- this is a secondary blocking signal, not the only one -- but confirm
    the actual hit rate on real rows, not this docstring.
    """
    segments = [s.strip() for s in address_norm.split(",") if s.strip()]
    if not segments:
        return ""
    if pin_code:
        for i, seg in enumerate(segments):
            if pin_code in seg:
                same_segment_city = _DIGITS_RE.sub("", seg).strip()
                prev_looks_like_street = i > 0 and bool(re.match(r"^\d", segments[i - 1]))
                if same_segment_city and (prev_looks_like_street or i == 0):
                    return same_segment_city
                return segments[i - 1] if i > 0 else ""
    return segments[-2] if len(segments) >= 2 else ""


def normalize_address_fields(raw_address: str) -> Dict[str, str]:
    text = raw_address or ""
    text = _NULL_TOKEN_RE.sub(" ", text)
    text = text.lower()
    pin_code = extract_postal_code(text)
    for pattern, replacement in _COMPILED_ADDRESS_ABBR:
        text = pattern.sub(replacement, text)
    # keep commas (city/state segmentation depends on them); drop other punctuation
    text = common.strip_punctuation(text, keep_chars=",")
    text = re.sub(r"\s*,\s*", ", ", text)
    text = common.collapse_ws(text)
    city_guess = guess_city(text, pin_code)
    return {"address_norm": text, "pin_code": pin_code, "city_guess": city_guess}


# --------------------------------------------------------------------------------------- driver

_OUTPUT_COLS = [
    "entity_id", "business_name", "business_address", "country",
    "name_norm", "legal_name_norm", "trade_name_norm", "has_dba",
    "address_norm", "pin_code", "city_guess", "name_script", "address_script",
]


def normalize_row(row) -> tuple:
    entity_id, raw_name, raw_address, country = row
    raw_name = raw_name or ""
    raw_address = raw_address or ""
    country = (country or "").strip()
    fields = {
        "entity_id": entity_id,
        "business_name": raw_name,
        "business_address": raw_address,
        "country": country,
        **normalize_name_fields(raw_name, country),
        **normalize_address_fields(raw_address),
        "name_script": common.detect_script(raw_name),
        "address_script": common.detect_script(raw_address),
    }
    return tuple(fields[c] for c in _OUTPUT_COLS)


def normalize_frame(df: pd.DataFrame, label: str = "", workers: int = None) -> pd.DataFrame:
    """Row-parallel across `workers` processes (default: every CPU core). Pure-Python regex work
    like this is CPU-bound and doesn't benefit from a GPU; using all cores is the real speedup."""
    workers = workers or os.cpu_count() or 1
    rows = list(zip(df["entity_id"], df["business_name"], df["business_address"], df["country"]))
    bar = tqdm(total=len(rows), desc=f"normalize {label}", unit="row", unit_scale=True, mininterval=1.0)
    if workers == 1 or len(rows) < 20_000:
        out = []
        for row in rows:
            out.append(normalize_row(row))
            bar.update(1)
    else:
        out = []
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            for result in pool.imap(normalize_row, rows, chunksize=2_000):
                out.append(result)
                bar.update(1)
    bar.close()
    return pd.DataFrame(out, columns=_OUTPUT_COLS)


def run(repo_root, splits, skip_existing: bool = False):
    """skip_existing: reuse normalized files already on disk (e.g. from a previous Kaggle session)
    instead of recomputing them. Only safe if they were produced by the current normalize.py."""
    for split in splits:
        for key in common.SOURCE_KEYS:
            if skip_existing and common.normalized_path(repo_root, split, key).exists():
                print(f"[normalize] {split}/{key}: reusing existing {common.normalized_path(repo_root, split, key)}")
                continue
            # one source at a time: holding all three raw splits at once needlessly doubles peak memory
            raw = common.read_source_tsv(common.require(
                common.dataset_dir(repo_root, split) / common.SOURCE_FILENAMES[split][key],
                "the dataset setup (notebook Section 0.5)"))
            print(f"[normalize] {split}/{key}: {len(raw)} rows")
            normalized = normalize_frame(raw, label=f"{split}/{key}")
            del raw
            out_path = common.write_normalized(normalized, repo_root, split, key)
            print(f"[normalize]   -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_repo_root_arg(parser)
    parser.add_argument("--split", action="append", choices=list(common.SPLITS),
                         help="Repeatable. Default: both train and test.")
    args = parser.parse_args()
    splits = args.split or list(common.SPLITS)
    run(args.repo_root, splits)


if __name__ == "__main__":
    main()
