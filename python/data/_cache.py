
"""
_cache.py
=========
Date-keyed on-disk cache with stale-fallback for data ingestion paths.
"""

import os
import glob
import pickle
import datetime
import warnings

from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"


def _normalize_date(valuation_date):
    if valuation_date is None:
        return datetime.date.today()
    if isinstance(valuation_date, datetime.datetime):
        return valuation_date.date()
    if isinstance(valuation_date, datetime.date):
        return valuation_date
    return datetime.date.fromisoformat(str(valuation_date))


def _cache_path(source_name, d):
    return CACHE_DIR / ("%s_%s.pkl" % (source_name, d.isoformat()))


def _find_latest_prior(source_name, d):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pattern = str(CACHE_DIR / ("%s_*.pkl" % source_name))
    candidates = []
    for path in glob.glob(pattern):
        stem = Path(path).stem
        date_part = stem[len(source_name) + 1:]
        try:
            cand_date = datetime.date.fromisoformat(date_part)
        except ValueError:
            continue
        if cand_date <= d:
            candidates.append((cand_date, path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def load_or_fetch(source_name, valuation_date, fetch_fn):
    d = _normalize_date(valuation_date)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(source_name, d)

    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)

    try:
        result = fetch_fn()
    except Exception as exc:
        fallback = _find_latest_prior(source_name, d)
        if fallback is None:
            raise
        warnings.warn(
            "[cache] %s fetch failed (%s); using stale cache %s"
            % (source_name, exc, os.path.basename(fallback))
        )
        with open(fallback, "rb") as f:
            return pickle.load(f)

    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result


def prune_cache(valuation_date) -> int:
    d = _normalize_date(valuation_date)
    if not CACHE_DIR.exists():
        return 0
    keep = d.isoformat()
    deleted = 0
    for path in CACHE_DIR.iterdir():
        if not path.is_file() or path.suffix != ".pkl":
            continue
        stem = path.stem
        idx = stem.rfind("_")
        if idx == -1:
            continue
        date_part = stem[idx + 1:]
        try:
            datetime.date.fromisoformat(date_part)
        except ValueError:
            continue
        if date_part != keep:
            path.unlink()
            deleted += 1
    return deleted
