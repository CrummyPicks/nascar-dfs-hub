"""NASCAR DFS Hub — Utility Functions."""

import re
import unicodedata

import pandas as pd
import numpy as np
from src.config import DK_FINISH_POINTS, FD_FINISH_POINTS, DRIVER_ALIASES


def parse_american_odds(value) -> int | None:
    """Parse a single American-odds value into a signed integer.

    Handles all the formats sportsbooks ship in:
        +350, -150, 350           -> 350, -150, 350
        " +350 ", "  -150  "      -> 350, -150  (whitespace tolerated)
        EVEN, even, Even, EV, PK  -> 100  (pick'em / evens = +100)
        None, "", "—", "N/A"      -> None

    Used by the manual-paste parser, the live-odds fetcher, the ownership
    model, and the accuracy backtest — anywhere a raw odds string needs to
    become a number.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.upper() in {"NONE", "NULL", "N/A", "NA", "—", "-", ""}:
        return None
    # Pick-em / evens: every book uses one of these spellings
    if s.upper() in {"EVEN", "EVENS", "EV", "PK", "PICK", "PICK'EM", "PICKEM"}:
        return 100
    # Strip optional + sign and any embedded whitespace, then parse
    cleaned = s.replace("+", "").strip()
    try:
        # Allow decimal odds files like "350.0" by going via float
        return int(float(cleaned))
    except (ValueError, TypeError):
        return None


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


def arp_finish_blend(arp, avg_finish, track_type: str = None) -> float:
    """Blend ARP and avg_finish into a single 'projected race result' signal.

    Weighting is track-type-aware because ARP and finish correlate
    differently across track types:

      - Superspeedway: big wrecks decouple ARP from finish. A driver who
        runs mid-pack (high ARP) but survives and finishes top-10 should
        be credited for the FINISH, not penalized for staying back to
        avoid the pileup. Weight: 35% ARP + 65% finish.
      - Road course: limited passing, so ARP tracks finish closely. Still
        lean slightly toward finish to account for pit-strategy outliers.
        Weight: 55% ARP + 45% finish.
      - Ovals (intermediate / short / etc): ARP is a strong pace proxy
        since lead-lap cars maintain track position. Weight: 65% ARP +
        35% finish.

    Returns avg_finish if arp is None (missing data).
    """
    if arp is None:
        return avg_finish
    if track_type == "superspeedway":
        # 40/60 — wrecks decouple running position from finish at supers,
        # so finish gets more weight. ARP still has 40% because finish
        # alone is noisy (one pileup can wreck a front-runner through no
        # fault of their own), but we lean clearly toward finish since
        # that's what ends up as DFS points. Combined with trimmed-mean
        # aggregation at supers (drop worst finish when >= 4 races), this
        # properly rewards drivers like Hocevar whose one-race wreck
        # otherwise poisoned their avg beyond race-pace reality.
        return arp * 0.40 + avg_finish * 0.60
    if track_type == "road":
        return arp * 0.55 + avg_finish * 0.45
    return arp * 0.65 + avg_finish * 0.35


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


def _nickname_canonical(normalized: str) -> str:
    """Expand a first-name nickname to its canonical long form.

    Input must already be normalized (lowercase, ASCII). Returns a version
    with the first token mapped through NICKNAME_MAP if applicable.
    Example: "nick sanchez" -> "nicholas sanchez".
    """
    from src.config import NICKNAME_MAP
    parts = normalized.split()
    if not parts:
        return normalized
    first = parts[0]
    canonical = NICKNAME_MAP.get(first)
    if canonical:
        parts[0] = canonical
    return " ".join(parts)


def _stripped_middle(normalized: str) -> str:
    """Drop single-character middle tokens — "jason m white" -> "jason white".

    Only drops tokens in the middle positions (not first, not last) that are
    exactly one character long. Catches the "middle initial is optional"
    pattern that shows up across NASCAR data sources.
    Input must already be normalized.
    """
    parts = normalized.split()
    if len(parts) <= 2:
        return normalized
    kept = [p for i, p in enumerate(parts)
            if i == 0 or i == len(parts) - 1 or len(p) > 1]
    return " ".join(kept)


def _match_keys(name: str) -> list:
    """Return all normalized keys a name could match against.

    First key is the primary normalized form. Additional keys cover
    systematic variants: nickname expansion (Nick -> Nicholas) and
    middle-initial stripping (Jason M White -> Jason White). When matching
    candidates, any shared key between name and candidate = match.
    """
    primary = normalize_driver_name(name)
    keys = [primary]
    # Nickname-expanded
    nc = _nickname_canonical(primary)
    if nc != primary:
        keys.append(nc)
    # Middle-initial stripped
    sm = _stripped_middle(primary)
    if sm != primary:
        keys.append(sm)
    # Also try combined: nickname + stripped middle
    nc_sm = _stripped_middle(nc)
    if nc_sm != primary and nc_sm not in keys:
        keys.append(nc_sm)
    return keys


def fuzzy_match_name(name: str, candidates: list, threshold: float = 0.75) -> str:
    """Find best fuzzy match for a driver name in a list of candidates.

    Matching passes (first hit wins):
      1. Exact normalized match (Suárez↔Suarez, periods stripped, aliases)
      2. Nickname equivalence (Nick↔Nicholas via NICKNAME_MAP)
      3. Middle-initial optional (Jason M White↔Jason White)
      4. Last-name match — only if that last name is unique in candidates
      5. SequenceMatcher fuzzy ratio (threshold default 0.75)
    """
    from difflib import SequenceMatcher

    if not name or not candidates:
        return None

    name_keys = set(_match_keys(name))
    # Pre-compute all match keys for each candidate (stored as a tuple of keys)
    norm_candidates = [(c, _match_keys(c)) for c in candidates]

    # Pass 1-3: any shared key between name and candidate = match
    for candidate, cand_keys in norm_candidates:
        if name_keys & set(cand_keys):
            return candidate

    # Pass 4: last-name match — only if unique
    primary_norm = normalize_driver_name(name)
    last_name = primary_norm.split()[-1] if primary_norm.split() else ""
    if last_name:
        last_name_matches = [
            (c, ck) for c, ck in norm_candidates
            if ck[0].split() and ck[0].split()[-1] == last_name
        ]
        if len(last_name_matches) == 1:
            return last_name_matches[0][0]

    # Pass 5: fuzzy SequenceMatcher
    best_match, best_score = None, 0.0
    for candidate, cand_keys in norm_candidates:
        score = SequenceMatcher(None, primary_norm, cand_keys[0]).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate

    return best_match if best_score >= threshold else None


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _last_name_token(name: str) -> str:
    """Return the last name of a driver, skipping over Jr/Sr/II/III suffixes.

    "Ricky Stenhouse Jr" -> "Stenhouse"
    "Kyle Busch"         -> "Busch"
    "A J Allmendinger"   -> "Allmendinger"
    """
    parts = [p for p in name.strip().split() if p]
    if not parts:
        return ""
    # Walk from the end, skipping suffix tokens
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() not in _NAME_SUFFIXES:
            return parts[i]
    return parts[-1]  # fallback: all tokens were suffixes (shouldn't happen)


def short_name(full_name: str, all_names: list = None) -> str:
    """Abbreviate a driver name for chart labels.

    Returns last name only ("Larson"), or "First Initial. Last" when there
    are duplicate last names in *all_names* ("K. Busch" vs "Ky. Busch").
    Handles Jr/Sr/II/III suffixes so "Ricky Stenhouse Jr" -> "Stenhouse"
    (not "Jr").
    """
    if not full_name or not isinstance(full_name, str):
        return str(full_name) if full_name else ""

    name = full_name.strip()
    parts = name.split()
    if len(parts) <= 1:
        return name

    last = _last_name_token(name)

    if all_names:
        # Check for duplicate last names — use _last_name_token so "Stenhouse Jr"
        # and "Stenhouse" both resolve to "Stenhouse" for comparison purposes.
        dup_lasts = [n for n in all_names
                     if isinstance(n, str) and _last_name_token(n) == last
                        and n.strip() != name]
        if dup_lasts:
            # Use enough of first name to disambiguate
            first = parts[0]
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
    one_dec_patterns = {"Avg Finish", "Avg Start",
                        "Avg Run", "Avg Running Position", "DFS Points", "DK Pts",
                        "FD Pts", "FD Points", "Proj Score", "Projected Score",
                        "Track Score", "Qual Score", "Practice Score",
                        "Weighted Score", "Score", "TH Avg Finish",
                        "TH Avg Start",
                        "TH_Avg DK", "TH_Best DK", "TH_Worst DK",
                        "TH_Avg Finish", "TH_Avg Start", "TH_Avg Run Pos",
                        "Avg DK", "Best DK", "Worst DK",
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
            if dtype_str not in ("object", "str", "string", "StringDtype"):
                result[col] = pd.to_numeric(result[col], errors="coerce").round(1)
        elif col_name in two_dec_patterns:
            if dtype_str not in ("object", "str", "string", "StringDtype"):
                result[col] = pd.to_numeric(result[col], errors="coerce").round(2)
        elif col_name in three_dec_patterns:
            if dtype_str not in ("object", "str", "string", "StringDtype"):
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
