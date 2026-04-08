"""NASCAR DFS Hub — Utility Functions."""

import unicodedata

import pandas as pd
import numpy as np
from src.config import DK_FINISH_POINTS, FD_FINISH_POINTS, DRIVER_ALIASES


def calc_dk_points(finish, start, laps_led, fastest_laps):
    """Calculate DraftKings NASCAR Classic DFS points."""
    try:
        place_pts = DK_FINISH_POINTS.get(int(finish), 0)
        diff_pts = (int(start) - int(finish)) * 1.0
        led_pts = int(laps_led) * 0.25
        fl_pts = int(fastest_laps) * 0.45
        return round(place_pts + diff_pts + led_pts + fl_pts, 2)
    except (ValueError, TypeError):
        return 0.0


def calc_fd_points(finish, start, laps_led, fastest_laps):
    """Calculate FanDuel NASCAR DFS points."""
    try:
        place_pts = FD_FINISH_POINTS.get(int(finish), 0)
        diff_pts = (int(start) - int(finish)) * 0.5
        led_pts = int(laps_led) * 0.1
        fl_pts = int(fastest_laps) * 0.5
        return round(place_pts + diff_pts + led_pts + fl_pts, 2)
    except (ValueError, TypeError):
        return 0.0


def normalize_driver_name(name: str) -> str:
    """Normalize a driver name for matching.

    Steps: lowercase, Unicode→ASCII folding (Suárez→suarez), strip periods,
    remove common suffixes (Jr/Sr/III), collapse whitespace, apply aliases.
    """
    if not name:
        return ""
    name = name.strip().lower()
    # Unicode → ASCII folding (handles Suárez, Préost, etc.)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    # Strip all periods (A.J. → AJ, Jr. → Jr)
    name = name.replace(".", "")
    # Collapse extra whitespace
    name = " ".join(name.split())
    # Remove common suffixes
    for suffix in [" jr", " sr", " iii", " ii", " iv"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
            break
    # Apply alias mapping
    alias = DRIVER_ALIASES.get(name)
    if alias:
        name = alias
    return name


def fuzzy_match_name(name: str, candidates: list, threshold: float = 0.75) -> str:
    """Find best fuzzy match for a driver name in a list of candidates.

    Uses normalized names (Unicode-folded, period-stripped, alias-resolved)
    and a last-name shortcut only when the last name is unique in the candidate list.
    """
    from difflib import SequenceMatcher

    if not name or not candidates:
        return None

    norm = normalize_driver_name(name)
    best_match, best_score = None, 0.0

    # Pre-compute normalized candidates
    norm_candidates = [(c, normalize_driver_name(c)) for c in candidates]

    # Pass 1: exact normalized match
    for candidate, norm_c in norm_candidates:
        if norm == norm_c:
            return candidate

    # Pass 2: last-name match — only if last name is unique among candidates
    last_name = norm.split()[-1] if norm.split() else ""
    if last_name:
        last_name_matches = [
            (c, nc) for c, nc in norm_candidates
            if nc.split()[-1] == last_name
        ]
        if len(last_name_matches) == 1:
            return last_name_matches[0][0]

    # Pass 3: fuzzy SequenceMatcher
    for candidate, norm_c in norm_candidates:
        score = SequenceMatcher(None, norm, norm_c).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate

    return best_match if best_score >= threshold else None


def short_name(full_name: str, all_names: list = None) -> str:
    """Abbreviate a driver name for chart labels.

    Returns last name only ("Larson"), or "First Initial. Last" when there
    are duplicate last names in *all_names* ("K. Busch" vs "Ky. Busch").
    If car number is provided as "#9 Chase Elliott", strips it and adds back.
    """
    if not full_name or not isinstance(full_name, str):
        return str(full_name) if full_name else ""

    name = full_name.strip()
    parts = name.split()
    if len(parts) <= 1:
        return name

    last = parts[-1]

    if all_names:
        # Check for duplicate last names
        dup_lasts = [n for n in all_names
                     if isinstance(n, str) and n.strip().split()[-1] == last and n.strip() != name]
        if dup_lasts:
            # Use enough of first name to disambiguate
            first = parts[0]
            # Check if single initial is enough
            initials_needed = 1
            for dup in dup_lasts:
                dup_first = dup.strip().split()[0]
                while (initials_needed < len(first) and initials_needed < len(dup_first)
                       and first[:initials_needed].lower() == dup_first[:initials_needed].lower()):
                    initials_needed += 1
            abbrev = first[:max(1, initials_needed)]
            return f"{abbrev}. {last}"

    return last


def short_name_series(names: list) -> list:
    """Abbreviate a list of driver names, handling duplicates automatically."""
    return [short_name(n, names) for n in names]


def build_norm_lookup(mapping: dict) -> dict:
    """Build a {normalized_driver_name: value} dict from {display_name: value}."""
    return {normalize_driver_name(k): v for k, v in mapping.items()}


def fuzzy_get(name: str, mapping: dict, norm_cache: dict = None):
    """Look up a driver in a dict using normalized name matching.

    Args:
        name: Driver name to look up.
        mapping: Original {display_name: value} dict.
        norm_cache: Optional pre-built {normalized_name: value} dict from
                    build_norm_lookup() — avoids rebuilding each call.
    """
    if name in mapping:
        return mapping[name]
    norm = normalize_driver_name(name)
    if norm_cache is None:
        norm_cache = build_norm_lookup(mapping)
    return norm_cache.get(norm)


def fuzzy_merge(left: pd.DataFrame, right: pd.DataFrame, on: str = "Driver",
                how: str = "left", right_cols: list = None) -> pd.DataFrame:
    """Merge two DataFrames using normalized driver name matching.

    Normalizes the *on* column on both sides to find matches, then performs
    a standard pandas merge.  Falls back to fuzzy_match_name() when
    normalization alone doesn't find a match.
    """
    right_dedup = right.drop_duplicates(on)
    if right_cols:
        keep = [on] + [c for c in right_cols if c in right_dedup.columns and c != on]
        right_dedup = right_dedup[keep]

    # Build norm→original mapping for right side
    right_norm = {}
    for n in right_dedup[on]:
        nn = normalize_driver_name(n)
        if nn not in right_norm:
            right_norm[nn] = n

    right_names = list(right_dedup[on])

    def _match(name):
        if name in right_names:
            return name
        nn = normalize_driver_name(name)
        if nn in right_norm:
            return right_norm[nn]
        # Expensive fallback — only when normalization fails
        return fuzzy_match_name(name, right_names)

    left = left.copy()
    left["_merge_key"] = left[on].map(_match)
    right_dedup = right_dedup.rename(columns={on: "_merge_key"})
    result = left.merge(right_dedup, on="_merge_key", how=how)
    result = result.drop(columns=["_merge_key"])
    return result


def int_col(series: pd.Series) -> pd.Series:
    """Convert a pandas Series to nullable integer type, safely handling any input."""
    numeric = pd.to_numeric(series, errors="coerce")
    # Force to float64 first to avoid object-dtype cast issues
    numeric = numeric.astype("float64")
    return numeric.astype("Int64")


def format_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply smart number formatting to a DataFrame for display.
    Rounds columns based on their name/type to remove excessive decimals.
    """
    result = df.copy()

    # Exact integer columns (no decimals)
    int_patterns = {"Rank", "Races", "Wins", "Top 5", "Top 10", "Top 20",
                    "Laps Led", "Fastest Laps", "Fast Laps", "DNF", "Laps",
                    "TH Races", "TH Wins", "TH T5", "TH T10", "TH T20",
                    "TH Laps Led", "TH DNF", "TH Best", "TH Worst",
                    "TH_Races", "TH_Wins", "TH_T5", "TH_T10", "TH_T20",
                    "TH_Laps Led", "TH_DNF",
                    "GFS Races", "Finish Position", "Start", "Qual",
                    "Qualifying Position", "Finish", "Pos Diff",
                    "Position Differential", "Projected Finish",
                    "Proj Laps Led", "Proj Fast Laps", "Count",
                    "Best Finish", "Worst Finish", "DK Salary",
                    "Qual Pos", "Win Odds", "Top 3 Odds", "Top 5 Odds", "Top 10 Odds"}

    # 1-decimal columns
    one_dec_patterns = {"Avg Finish", "Avg Start", "Avg Rating", "Rating",
                        "Avg Run", "Avg Running Position", "DFS Points", "DK Pts",
                        "FD Pts", "FD Points", "Proj Score", "Projected Score",
                        "Track Score", "Qual Score", "Practice Score",
                        "Weighted Score", "Score", "TH Avg Finish",
                        "TH Avg Start", "TH Rating",
                        "TH_Avg Finish", "TH_Avg Start", "TH_Rating",
                        "Avg DFS", "Best DFS",
                        "Worst DFS", "GFS Avg DK Pts", "GFS Avg FD Pts",
                        "Avg Laps Led", "Avg Fastest Laps", "Avg Fast Laps",
                        "Penn Rank", "Proj DK", "Proj Finish",
                        "Finish Pts", "Diff Pts", "Led Pts", "FL Pts",
                        "Track", "Track Type", "Avg DK", "Avg FD",
                        "Avg_Proj", "Avg_Value", "Impl %"}

    # 2-decimal columns
    two_dec_patterns = {"Value", "DFS Value"}

    # 3-decimal columns (lap times)
    three_dec_patterns = {"Overall Avg", "Best Lap", "Best Lap Time",
                          "5 Lap", "10 Lap", "15 Lap", "20 Lap", "25 Lap", "30 Lap",
                          "Qual Speed", "Best Lap Speed"}

    for col in result.columns:
        # Handle MultiIndex columns
        col_name = col[-1] if isinstance(col, tuple) else col
        dtype_str = str(result[col].dtype)

        if col_name in int_patterns or col_name.endswith("Rank"):
            if dtype_str not in ("object", "str", "string", "StringDtype"):
                try:
                    result[col] = pd.to_numeric(result[col], errors="coerce").astype("float64").astype("Int64")
                except (TypeError, ValueError):
                    pass  # Skip if conversion fails
        elif col_name in one_dec_patterns:
            result[col] = pd.to_numeric(result[col], errors="coerce").round(1)
        elif col_name in two_dec_patterns:
            result[col] = pd.to_numeric(result[col], errors="coerce").round(2)
        elif col_name in three_dec_patterns:
            result[col] = pd.to_numeric(result[col], errors="coerce").round(3)

    return result


def safe_fillna(df: pd.DataFrame, fill_value="") -> pd.DataFrame:
    """Fill NaN values safely for Streamlit display.
    Numeric columns stay numeric (converted to float64 which handles NaN natively)
    so that Streamlit column sorting works correctly.
    Non-numeric columns get fill_value (empty string by default).
    """
    result = df.copy()
    for col in result.columns:
        dtype_str = str(result[col].dtype)
        if dtype_str in ("Int64", "Int32", "Int16", "Int8"):
            # Convert nullable int to float64 — preserves NaN and numeric sorting
            result[col] = result[col].astype("float64")
        elif dtype_str.startswith("float"):
            # Already numeric, leave as-is (NaN displays as blank in Streamlit)
            pass
        else:
            result[col] = result[col].fillna(fill_value)
    return result
