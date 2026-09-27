#!/usr/bin/env python3
"""
Preprocessing for the Amazon ML Challenge 2026 entity-resolution task, designed for
retrieval with a multilingual embedding model.

For every record it produces
  * embedding text  : a lightly cleaned, consistently written "name | address" string
                      (junk removed, abbreviations expanded, Indic words translated,
                      legal form removed) and an address-only string;
  * matcher fields  : a few structured values the embedding cannot resolve on its own
                      (legal form, house number, unit, address numbers, flags).

Lookup tables are learned from the provided data only:
  * Indic word -> English word      (aligned true training pairs)
  * state code / native script -> state name (aligned true training pairs, per country)
  * French department -> region     (co-occurrence of cities in the test files, no labels)
Hand-written tables below cover closed vocabularies: street types, unit words,
directions, ordinals, legal forms and a few common renames.

Output mirrors the input layout:
  <out>/train/train_source{1,2,3}.tsv, <out>/train/train_ground_truth.tsv (copied),
  <out>/test/test_source{1,2,3}.tsv, <out>/resources/*.json
Each source file keeps its 4 original columns and adds the derived columns (see OUT_COLS).

Usage (from the project root):
    python code/preprocess.py                      # learn tables + process all files
    python code/preprocess.py --workers 40
    python code/preprocess.py --demo               # print before/after for sample rows
"""
import argparse
import collections
import csv
import json
import multiprocessing as mp
import os
import re
import shutil
import time
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd
from rapidfuzz.distance import DamerauLevenshtein

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "student_resource", "dataset")
OUT_DIR = os.path.join(ROOT, "student_resource", "dataset_processed")
SPLITS = {"train": (1, 2, 3), "test": (1, 2, 3)}

OUT_COLS = [
    "name_clean",      # full cleaned name (legal words in canonical form)
    "name_core",       # name without legal form / honorifics - used for embedding
    "legal_form",      # canonical legal form tokens, sorted (e.g. "llc", "ltd pvt", "sarl")
    "dba_other",       # the other half of an "X dba Y" name (Y is kept as the main name)
    "addr_clean",      # cleaned, expanded address, components joined by ", "
    "house_no",        # best guess of the street / house number, full token ("5-d", "15/64")
    "house_base",      # leading number of house_no only ("5-d" -> "5", "209-211" -> "209")
    "unit",            # unit / apartment / suite / floor / PO box value
    "addr_numbers",    # every number-like address token, space separated
    "is_indic",        # name had Indic script in the raw data
    "is_domain",       # name was a domain or @/# handle
    "has_dba",         # name had a dba / trading-as part
    "addr_empty",      # address empty after cleaning
    "embed_text",      # "name_core | addr_clean"  -> multilingual embedding
    "embed_text_addr", # addr_clean                -> address-only embedding
]

# ======================================================================================
# Hand-written tables (closed vocabularies). Keys are lower-case, accent-folded tokens.
# ======================================================================================

# --- Street types -------------------------------------------------------------------
US_STREET = {
    "rd": "road", "st": "street", "str": "street", "dr": "drive", "drv": "drive", "ave": "avenue",
    "av": "avenue", "ln": "lane", "ct": "court", "crt": "court", "cir": "circle", "pl": "place",
    "blvd": "boulevard", "trl": "trail", "hwy": "highway", "ter": "terrace", "terr": "terrace",
    "pkwy": "parkway", "pky": "parkway", "cv": "cove", "sq": "square", "rdg": "ridge", "pt": "point",
    "ft": "fort", "mt": "mount", "mtn": "mountain", "ctr": "center", "cres": "crescent",
    "xing": "crossing", "expy": "expressway", "fwy": "freeway", "jct": "junction", "holw": "hollow",
    "hts": "heights", "spg": "spring", "spgs": "springs", "vly": "valley", "vw": "view", "aly": "alley",
    "bnd": "bend", "brg": "bridge", "byp": "bypass", "cswy": "causeway", "crk": "creek", "frk": "fork",
    "grv": "grove", "hbr": "harbor", "lk": "lake", "lndg": "landing", "mdw": "meadow", "mdws": "meadows",
    "plz": "plaza", "rte": "route", "rt": "route", "shr": "shore", "sta": "station", "tpke": "turnpike",
    "trce": "trace", "vlg": "village", "pass": "pass", "run": "run", "way": "way",
}
INDIA_STREET = {
    "rd": "road", "st": "street", "ngr": "nagar", "opp": "opposite", "nr": "near", "bldg": "building",
    "apt": "apartment", "apts": "apartments", "flr": "floor", "fl": "floor", "plt": "plot",
    "sec": "sector", "sect": "sector", "ph": "phase", "extn": "extension", "ext": "extension",
    "col": "colony", "clny": "colony", "soc": "society", "hsg": "housing", "mkt": "market",
    "stn": "station", "dist": "district", "distt": "district", "dt": "district", "tq": "taluk",
    "tal": "taluk", "tk": "taluk", "vill": "village", "vil": "village", "indl": "industrial",
    "blk": "block", "jn": "junction", "jct": "junction", "mg": "mg", "c/o": "care of",
    "g/f": "ground floor", "gf": "ground floor", "f/f": "first floor", "ff": "first floor",
    "s/f": "second floor", "sf": "second floor", "hno": "house no", "sno": "survey no",
    "pno": "plot no", "fno": "flat no", "aprts": "apartments", "aprt": "apartment",
}
FRANCE_STREET = {
    "r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard",
    "blvd": "boulevard", "boul": "boulevard", "bvd": "boulevard", "all": "allee", "imp": "impasse",
    "ch": "chemin", "che": "chemin", "chem": "chemin", "pl": "place", "rte": "route", "fbg": "faubourg",
    "faub": "faubourg", "sq": "square", "crs": "cours", "qu": "quai", "res": "residence",
    "resid": "residence", "st": "saint", "ste": "sainte", "bat": "batiment", "esc": "escalier",
    "apt": "appartement", "zi": "zone industrielle", "za": "zone artisanale",
}
# --- Units, directions, ordinals ----------------------------------------------------
US_UNIT = {"apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor", "bldg": "building",
           "rm": "room", "spc": "space", "trlr": "trailer"}
DIRECTIONS = {"n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast",
              "nw": "northwest", "se": "southeast", "sw": "southwest"}
ORDINALS = {"first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
            "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
            "eleventh": "11th", "twelfth": "12th", "thirteenth": "13th", "fourteenth": "14th",
            "fifteenth": "15th", "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
            "nineteenth": "19th", "twentieth": "20th"}
# Multi-token India forms after punctuation is removed ("h.no" -> "h no").
INDIA_PHRASES = [(r"\bh no\b", "house no"), (r"\bs no\b", "survey no"), (r"\bp no\b", "plot no"),
                 (r"\bpl no\b", "plot no"), (r"\bf no\b", "flat no"), (r"\bsy no\b", "survey no"),
                 (r"\bd no\b", "door no")]
UNIT_KW = {"unit", "apartment", "suite", "floor", "flat", "shop", "room", "box", "office", "appartement",
           "batiment", "escalier", "space", "trailer"}
# --- Renames / synonyms (token level, per country) -------------------------------------
SYNONYMS = {
    "India": {"calcutta": "kolkata", "bombay": "mumbai", "madras": "chennai", "poona": "pune",
              "bengaluru": "bangalore", "gurugram": "gurgaon", "trivandrum": "thiruvananthapuram",
              "baroda": "vadodara", "pondicherry": "puducherry", "mysuru": "mysore",
              "mangaluru": "mangalore", "orissa": "odisha", "keralam": "kerala",
              "uttaranchal": "uttarakhand", "calicut": "kozhikode", "cochin": "kochi"},
}
# --- Names ----------------------------------------------------------------------------
NAME_ABBR = {"intl": "international", "mfg": "manufacturing", "svcs": "services", "svc": "service",
             "assoc": "associates", "assn": "association", "bros": "brothers", "mgmt": "management",
             "mgt": "management", "natl": "national", "univ": "university", "hosp": "hospital",
             "ctr": "center", "centre": "center", "govt": "government", "engg": "engineering",
             "inst": "institute", "dept": "department", "coop": "cooperative", "sté": "societe",
             "ste": "societe", "ets": "etablissements"}
HONORIFICS = {"the", "dr", "smt", "shri", "sri", "m/s", "ms", "mr", "mrs", "mister"}
LEGAL = {  # token -> canonical legal token
    "common": {"inc": "inc", "incorporated": "inc", "llc": "llc", "corp": "corp", "corporation": "corp",
               "co": "co", "company": "co", "ltd": "ltd", "limited": "ltd", "pc": "pc", "pllc": "pllc",
               "plc": "plc", "lp": "lp", "llp": "llp", "group": "group", "holdings": "holdings",
               "pvt": "pvt", "private": "pvt", "opc": "opc"},
    "France": {"sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sci": "sci", "sa": "sa",
               "snc": "snc", "scop": "scop", "selarl": "selarl", "gie": "gie", "scm": "scm",
               "association": "association", "asso": "association"},
    "India": {"public": "public"},
}
DBA_RE = re.compile(r"\b(?:doing business as|d/b/a|dba|t/a|trading as)\b:?", re.I)
DOMAIN_RE = re.compile(r"^(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|in|net|org|co|fr|biz|info|io|us)(?:\.in)?$")
NULL_TOKENS = {"null", "none", "n/a", "na", "nan", "nil"}
LEET = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b"}

INDIC_RE = re.compile(r"[ऀ-෿]")
NUMLIKE_RE = re.compile(r"^(?:[a-z]{0,2}-?)?\d+(?:[a-z]|bis|ter)?(?:[/\-]\w+)*$")
ORD_NUM_RE = re.compile(r"^\d+(?:st|nd|rd|th)$")


# ======================================================================================
# Basic text helpers
# ======================================================================================
def fold(s):
    """NFKC; strip accents for non-Indic text (Indic vowel signs are combining marks!)."""
    s = unicodedata.normalize("NFKC", s)
    if INDIC_RE.search(s):
        return unicodedata.normalize("NFC", s)
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def join_dotted(s):
    """'l.l.c.' -> 'llc', 's.a.r.l' -> 'sarl' (before dots become spaces)."""
    return re.sub(r"\b(?:[a-z]\.){2,}[a-z]?\.?", lambda m: m.group().replace(".", ""), s)


# ======================================================================================
# Resource learning
# ======================================================================================
def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


def comps_raw(a):
    out = []
    for c in a.split(","):
        c = re.sub(r"\s+", " ", fold(c.strip().lower()))
        if c and c not in NULL_TOKENS:
            out.append(c)
    return out


def simple_tokens(s):
    return re.findall(r"[a-z0-9]+", join_dotted(fold(s.lower())))


def learn_resources(data_dir, n_pairs=3_000_000, seed=0):
    t0 = time.time()
    s1 = read_tsv(os.path.join(data_dir, "train", "train_source1.tsv")).set_index("entity_id")
    o = pd.concat([read_tsv(os.path.join(data_dir, "train", f"train_source{s}.tsv")) for s in (2, 3)]).set_index("entity_id")
    gt = read_tsv(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    p = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    p = p[p.m != ""]
    p = p.sample(min(n_pairs, len(p)), random_state=seed)
    A, B = s1.loc[p.source1_entity_id], o.loc[p.m]
    print(f"  loaded train pairs ({len(p):,}) in {time.time() - t0:.0f}s")

    # 1) Indic word -> English word, by position in names with equal token counts.
    wm = collections.defaultdict(collections.Counter)
    for a, b in zip(A.business_name.values, B.business_name.values):
        if not INDIC_RE.search(b):
            continue
        ta = [re.sub(r"[^\w&]", "", x) for x in fold(a.lower()).replace(".", " ").split()]
        tb = [re.sub(r"[^\w\u0900-\u0dff]", "", x) for x in unicodedata.normalize("NFC", b).split()]
        if len(ta) != len(tb):
            continue
        for x, y in zip(tb, ta):
            if INDIC_RE.search(x) and y:
                wm[x][y] += 1
    indic_map = {}
    for k, v in wm.items():
        w, n = v.most_common(1)[0]
        if n >= 3 and n / sum(v.values()) >= 0.5:
            indic_map[k] = w
    print(f"  Indic word map: {len(indic_map):,} words")

    # 2) State aliases per country: single differing, digit-free address components.
    pair = collections.defaultdict(collections.Counter)
    bcount = collections.defaultdict(collections.Counter)
    for c, a, b in zip(A.country.values, A.business_address.values, B.business_address.values):
        ca, cb = set(comps_raw(a)), set(comps_raw(b))
        da = [x for x in ca - cb if not re.search(r"\d", x)]
        db = [x for x in cb - ca if not re.search(r"\d", x)]
        for x in db:
            bcount[c][x] += 1
        if len(da) == 1 and len(db) == 1:
            pair[c][(db[0], da[0])] += 1
    is_code = lambda x: len(x) == 2 and x.isalpha()
    state_map = {}
    for c in pair:
        votes = collections.defaultdict(collections.Counter)
        for (b, a), n in pair[c].items():
            if n < 10 or n / bcount[c][b] < 0.2:
                continue
            if is_code(a) and not is_code(b) and not INDIC_RE.search(b):
                votes[a][b] += n              # code in S1 -> full name
            elif is_code(b) and not is_code(a):
                votes[b][a] += n              # code in S2/S3 -> full name
            elif INDIC_RE.search(b) and not is_code(a):
                votes[b][a] += n              # native script -> English name
        m = {k: v.most_common(1)[0][0] for k, v in votes.items()}
        # resolve chains, e.g. native -> 'orissa' and synonyms
        syn = SYNONYMS.get(c, {})
        state_map[c] = {k: syn.get(v, v) for k, v in m.items()}
        print(f"  state aliases {c}: {len(state_map[c])}")

    # 3) Vocabularies from S1 (the clean reference): names for leet repair / word
    #    segmentation, addresses for typo repair.
    name_vocab, addr_vocab = collections.Counter(), collections.Counter()
    for sp in ("train", "test"):
        d = read_tsv(os.path.join(data_dir, sp, f"{sp}_source1.tsv"))
        for n in d.business_name.values:
            name_vocab.update(simple_tokens(n))
        for a in d.business_address.values:
            addr_vocab.update(simple_tokens(a))
    name_vocab = {k: v for k, v in name_vocab.items() if v >= 3}
    addr_vocab = {k: v for k, v in addr_vocab.items() if v >= 3}
    print(f"  S1 vocab: {len(name_vocab):,} name tokens, {len(addr_vocab):,} address tokens")

    # 4) France (no labels): department -> region through shared cities.
    region_map = learn_region_map(data_dir, country="France")
    print(f"  France department -> region: {region_map}")
    print(f"  resources learned in {time.time() - t0:.0f}s")
    return {"indic_map": indic_map, "state_map": state_map, "name_vocab": name_vocab,
            "addr_vocab": addr_vocab, "region_map": {"France": region_map}}


def learn_region_map(data_dir, country="France", min_share=0.01):
    """Map admin components used only by S2/S3 (e.g. 'nord') to the one S1 uses ('hauts-de-france').

    Unsupervised: a city appears with the region in S1 and with the department in S2/S3,
    so the department inherits the region of the cities it co-occurs with.
    """
    def comps_for(src):
        d = read_tsv(os.path.join(data_dir, "test", f"test_source{src}.tsv"))
        d = d[d.country == country]
        return [[x for x in comps_raw(a) if not re.search(r"\d", x)] for a in d.business_address.values]

    c1 = comps_for(1)
    c23 = comps_for(2) + comps_for(3)
    if not c1 or not c23:
        return {}
    f1 = collections.Counter(x for cs in c1 for x in set(cs))
    f23 = collections.Counter(x for cs in c23 for x in set(cs))
    freq1 = {x for x, n in f1.items() if n / len(c1) >= min_share}
    # S1 regions = frequent S1 components that co-occur with many different frequent components.
    nb = collections.defaultdict(set)
    for cs in c1:
        for x in cs:
            if x in freq1:
                nb[x].update(y for y in cs if y in freq1 and y != x)
    regions = {x for x in freq1 if len(nb[x]) >= 3}
    city_region = {}
    for cs in c1:
        r = [x for x in cs if x in regions]
        for x in cs:
            if x in freq1 and x not in regions and len(r) == 1:
                city_region.setdefault(x, collections.Counter())[r[0]] += 1
    city_region = {c: v.most_common(1)[0][0] for c, v in city_region.items()}
    depts = [x for x, n in f23.items() if n / len(c23) >= min_share and f1.get(x, 0) / len(c1) < 0.001]
    out = {}
    for dpt in depts:
        votes = collections.Counter()
        for cs in c23:
            if dpt in cs:
                for x in cs:
                    if x in city_region:
                        votes[city_region[x]] += 1
        if votes:
            r, n = votes.most_common(1)[0]
            if n / sum(votes.values()) >= 0.8:
                out[dpt] = r
    return out


# ======================================================================================
# Record-level processing (uses module-level RES, set before workers fork)
# ======================================================================================
RES = None


def set_resources(res):
    global RES
    RES = res
    word_segment.cache_clear()
    repair_token.cache_clear()
    _fix_typo.cache_clear()


@lru_cache(maxsize=200_000)
def word_segment(s):
    """Split a joined string into S1 name words ('firstnetworks' -> 'first networks')."""
    vocab = RES["name_vocab"]
    n = len(s)
    best = [None] * (n + 1)
    best[0] = (0.0, [])
    for i in range(1, n + 1):
        for j in range(max(0, i - 20), i):
            w = s[j:i]
            if best[j] is None or (len(w) < 2 and w not in ("a",)) or w not in vocab or vocab[w] < 20:
                continue
            score = best[j][0] + np.log(vocab[w]) - 8.0      # penalty per word -> fewer words
            if best[i] is None or score > best[i][0]:
                best[i] = (score, best[j][1] + [w])
    if best[n] is None or len(best[n][1]) > 4:
        return s
    return " ".join(best[n][1])


def segment_handle(x):
    """Domain / handle body -> words; retry with leet digits replaced ('ph0enix...')."""
    seg = word_segment(x)
    if seg == x and re.search(r"\d", x):
        alt = "".join(LEET.get(ch, ch) for ch in x)
        seg2 = word_segment(alt)
        if seg2 != alt:
            return seg2
    return seg


@lru_cache(maxsize=500_000)
def repair_token(t):
    """Fix leet-style digits ('hea1th' -> 'health', '6roup' -> 'group', 'lnc' -> 'inc')."""
    vocab = RES["name_vocab"]
    own = vocab.get(t, 0)
    if own >= 50 or len(t) < 3 or ORD_NUM_RE.match(t) or t.isdigit():
        return t
    cands = []
    if re.search(r"\d", t) and re.search(r"[a-z]", t) and sum(ch.isalpha() for ch in t) >= len(t) - 2:
        cands.append("".join(LEET.get(ch, ch) for ch in t))                       # 1 -> l
        cands.append("".join({"1": "i"}.get(ch, LEET.get(ch, ch)) for ch in t))  # 1 -> i
    if t[0] == "l":
        cands.append("i" + t[1:])
    for c in cands:
        if vocab.get(c, 0) >= 50 and vocab[c] > 20 * own:
            return c
    return t


def clean_name(raw, country):
    """Return name_clean, name_core, legal_form, dba_other, is_domain, has_dba."""
    s = raw
    s = re.sub(r"\s*\|.*$", "", s)                              # "| www.x.com"
    s = re.sub(r"(?i)\(?\bid\s*[:#]?\s*\d+\)?", " ", s)           # "(ID: 47290)"
    s = re.sub(r"\s[-–]\s*\+?\d[\d\s\-]{6,}$", " ", s)            # "- 9876543210"
    s = re.sub(r"\s#\d{3,}$", " ", s)                             # "#48213"
    s = fold(s).lower()
    has_dba = 0
    dba_other = ""
    parts = DBA_RE.split(s)
    if len(parts) > 1 and parts[-1].strip():
        has_dba = 1
        dba_other, s = " ".join(parts[:-1]).strip(), parts[-1]
    s = join_dotted(s)
    s = s.replace("&", " and ")
    # domains / handles on the whole-token level (before dots are removed)
    is_domain = 0
    toks = []
    for t in s.split():
        tt = t.strip("*#@.,;:!?()[]{}<>~=\"'")
        m = DOMAIN_RE.match(tt)
        if m:
            is_domain = 1
            toks.append(segment_handle(m.group(1).replace("-", "")))
        elif re.match(r"^[#@][a-z0-9]{4,}$", t) and len(s.split()) == 1:
            is_domain = 1
            toks.append(segment_handle(t[1:]))
        else:
            toks.append(t)
    s = " ".join(toks)
    s = re.sub(r"(?<=[a-z])'(?=[a-z])", "", s)                   # schulman's -> schulmans
    s = re.sub(r"[^\w\s+/ऀ-෿]", " ", s)                 # punctuation, brackets -> space
    s = re.sub(r"(?<!\w)/|/(?!\w)", " ", s)
    toks = s.split()
    # Indic words -> English, digit repair, name abbreviations
    im = RES["indic_map"]
    out = []
    for t in toks:
        if INDIC_RE.search(t):
            t = im.get(unicodedata.normalize("NFC", t), t)
        else:
            t = repair_token(t)
        t = NAME_ABBR.get(t, t)
        out.extend(t.split())
    # drop leading honorifics (keep at least one token)
    while len(out) > 1 and out[0] in HONORIFICS:
        out.pop(0)
    legal = dict(LEGAL["common"])
    legal.update(LEGAL.get(country, {}))
    canon = [legal.get(t, t) for t in out]
    legal_form = " ".join(sorted({legal[t] for t in out if t in legal}))
    core = [t for t in out if t not in legal]
    while core and core[-1] in ("and", "of", "+"):                   # "terminus & co" -> "terminus"
        core.pop()
    while core and core[0] in ("and", "of", "+"):
        core.pop(0)
    name_clean = " ".join(canon)
    name_core = " ".join(core) if core else name_clean
    return name_clean, name_core, legal_form, dba_other, is_domain, has_dba


def _street_table(country):
    return {"US": US_STREET, "India": INDIA_STREET, "France": FRANCE_STREET}.get(country, US_STREET)


@lru_cache(maxsize=8)
def _street_full(country):
    return tuple(sorted(set(v for v in _street_table(country).values() if " " not in v and len(v) >= 4)))


@lru_cache(maxsize=200_000)
def _fix_typo(t, country):
    """One edit (incl. transposition) away from a known street word -> that word."""
    if len(t) < 4 or not t.isalpha() or RES["addr_vocab"].get(t, 0) >= 50:
        return t
    for w in _street_full(country):
        if abs(len(w) - len(t)) <= 1 and DamerauLevenshtein.distance(t, w) <= 1:
            return w
    return t


def clean_address(raw, country):
    """Return addr_clean, house_no, unit, addr_numbers."""
    street = _street_table(country)
    states = RES["state_map"].get(country, {})
    regions = RES["region_map"].get(country, {})
    syn = SYNONYMS.get(country, {})
    comps = []
    for c in raw.split(","):
        c = fold(c.strip()).lower()
        c = re.sub(r"\s+", " ", c).strip()
        if not c or c in NULL_TOKENS:
            continue
        # whole-component tables first: state codes / native names / departments
        if c in states:
            comps.append(states[c]); continue
        if c in regions:
            comps.append(regions[c]); continue
        c = join_dotted(c)
        if country == "France":
            c = re.sub(r"(?<=\d)\s*(bis|ter)\b", r"\1", c)          # "5 bis" -> "5bis"
        c = c.replace("#", " ")
        c = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", c)                  # dots, except inside numbers
        c = re.sub(r"[^\w\s/\-ऀ-෿]", " ", c)
        c = re.sub(r"(?<=\d)-(?=\s|$)", " ", c)                    # "2813-" -> "2813"
        c = re.sub(r"(?:(?<=\s)|^)-(?=\w)", " ", c)                    # "-1504" -> "1504"
        c = re.sub(r"\bno[-:]?(?=\d)", "no ", c)                        # "no-8-61" -> "no 8-61"
        if country == "India":
            for pat, rep in INDIA_PHRASES:
                c = re.sub(pat, rep, c)
        toks = [t.strip("-/") for t in c.split() if t not in NULL_TOKENS and t.strip("-/")]
        if not toks:
            continue
        starts_num = bool(re.match(r"^\d", toks[0]))
        out = []
        for i, t in enumerate(toks):
            last = i == len(toks) - 1
            if re.search(r"\d", t):
                t = re.sub(r"(?<!\d)0+(?=\d)", "", t)                   # "011265"->"11265", "g-009"->"g-9"
            elif t in ORDINALS:
                t = ORDINALS[t]
            elif country == "France":
                t = street.get(t, t)
            elif t in ("st", "saint") and country != "India":
                # "st"/"saint" at the end of a numbered street -> street; at the start of a place -> saint
                t = "street" if (i > 0 and (last or starts_num)) else "saint"
            elif country == "US" and t in DIRECTIONS and starts_num and (i == 1 or last):
                t = DIRECTIONS[t]
            elif country in ("US",) and t in US_UNIT:
                t = US_UNIT[t]
            elif t in street:
                t = street[t]
            else:
                t = _fix_typo(t, country)
            t = syn.get(t, t)
            out.extend(t.split())
        comps.append(" ".join(out))
    addr_clean = ", ".join(comps)

    # structured fields
    unit, nums = "", []
    house_kw = street_first = first_any = bare = first_all = ""
    is_num = lambda x: bool(NUMLIKE_RE.match(x)) and not ORD_NUM_RE.match(x)
    for comp in comps:
        toks = comp.split()
        skip = set()
        if len(toks) == 1 and is_num(toks[0]):          # bare "# 3" / "5/2" component
            bare = bare or toks[0]
            skip.add(0)
        for i, t in enumerate(toks):
            if t not in UNIT_KW:
                continue
            if t == "floor":                             # "1st floor", "ground floor", "floor 2"
                val = toks[i - 1] if i > 0 and (is_num(toks[i - 1]) or ORD_NUM_RE.match(toks[i - 1]) or toks[i - 1] == "ground") else ""
                if not val and i + 1 < len(toks) and is_num(toks[i + 1]):
                    val = toks[i + 1]; skip.add(i + 1)
                unit = unit or val
                continue
            j = i + 1
            while j < len(toks) and (toks[j] in UNIT_KW or toks[j] == "no"):
                j += 1
            if j < len(toks) and (is_num(toks[j]) or len(toks[j]) <= 2):
                unit = unit or toks[j]
                skip.add(j)
        for i, t in enumerate(toks):
            if not is_num(t):
                continue
            nums.append(t)
            first_all = first_all or t
            if i in skip:
                continue
            prev = " ".join(toks[max(0, i - 2):i])
            if not house_kw and prev in ("house no", "door no", "plot no"):
                house_kw = t
            if not street_first and i == 0 and any(x.isalpha() and x not in UNIT_KW for x in toks[1:]):
                street_first = t
            first_any = first_any or t
    if country == "India":
        # India swaps the prefix word (house / flat / shop / door / plot no) between sources,
        # so the first number in the address is the most stable choice.
        return addr_clean, first_all, unit, " ".join(nums)
    house = house_kw or street_first
    if bare:                                           # "# 3, 4610 Southern Pkwy" -> unit 3
        if house:
            unit = unit or bare
        else:                                          # "5/2, Russel Street" -> house 5/2
            house = bare
    house = house or first_any
    return addr_clean, house, unit, " ".join(nums)


def process_row(args):
    eid, name, addr, country = args
    name_clean, name_core, legal_form, dba_other, is_domain, has_dba = clean_name(name, country)
    addr_clean, house, unit, nums = clean_address(addr, country)
    m = re.search(r"\d+", house)
    house_base = m.group() if m else ""
    embed = f"{name_core} | {addr_clean}" if addr_clean else name_core
    return (name_clean, name_core, legal_form, dba_other, addr_clean, house, house_base, unit, nums,
            int(bool(INDIC_RE.search(name))), is_domain, has_dba, int(addr_clean == ""), embed, addr_clean)


def _process_chunk(rows):
    out = [process_row(r) for r in rows]
    # tabs/newlines must never reach the TSV
    return [tuple(x.replace("\t", " ").replace("\n", " ") if isinstance(x, str) else x for x in r) for r in out]


def process_file(src, dst, pool, chunk=20_000):
    t = time.time()
    df = read_tsv(src)
    rows = list(zip(df.entity_id, df.business_name, df.business_address, df.country))
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    res = [r for part in pool.imap(_process_chunk, chunks) for r in part]
    out = pd.DataFrame(res, columns=OUT_COLS)
    df = pd.concat([df.reset_index(drop=True), out], axis=1)
    df.to_csv(dst, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
    print(f"  {os.path.relpath(dst, ROOT)}: {len(df):,} rows in {time.time() - t:.0f}s")
    return df


# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--relearn", action="store_true", help="re-learn resources even if cached")
    ap.add_argument("--demo", action="store_true", help="print before/after for sample rows and exit")
    a = ap.parse_args()

    res_dir = os.path.join(a.out, "resources")
    os.makedirs(res_dir, exist_ok=True)
    res_path = os.path.join(res_dir, "resources.json")
    if os.path.exists(res_path) and not a.relearn:
        res = json.load(open(res_path))
        print(f"loaded resources from {os.path.relpath(res_path, ROOT)}")
    else:
        print("learning resources ...")
        res = learn_resources(a.data_dir)
        json.dump(res, open(res_path, "w"), ensure_ascii=False)
        for k in ("indic_map", "state_map", "region_map"):   # small readable copies
            json.dump(res[k], open(os.path.join(res_dir, f"{k}.json"), "w"), ensure_ascii=False, indent=1)
    set_resources(res)

    if a.demo:
        for sp, s in (("train", 1), ("train", 2), ("train", 3), ("test", 2), ("test", 3)):
            d = read_tsv(os.path.join(a.data_dir, sp, f"{sp}_source{s}.tsv")).sample(6, random_state=0)
            for r in d.itertuples(index=False):
                o = process_row((r.entity_id, r.business_name, r.business_address, r.country))
                print(f"\n[{sp} S{s} {r.country}] {r.business_name} | {r.business_address}")
                print("   embed :", o[OUT_COLS.index("embed_text")])
                print("   fields:", {k: o[OUT_COLS.index(k)] for k in ("legal_form", "house_no", "house_base", "unit", "addr_numbers")})
        return

    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers) as pool:
        for sp, srcs in SPLITS.items():
            os.makedirs(os.path.join(a.out, sp), exist_ok=True)
            for s in srcs:
                process_file(os.path.join(a.data_dir, sp, f"{sp}_source{s}.tsv"),
                             os.path.join(a.out, sp, f"{sp}_source{s}.tsv"), pool)
    shutil.copy(os.path.join(a.data_dir, "train", "train_ground_truth.tsv"),
                os.path.join(a.out, "train", "train_ground_truth.tsv"))
    print("copied train_ground_truth.tsv")


if __name__ == "__main__":
    main()
