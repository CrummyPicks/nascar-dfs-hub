"""NASCAR DFS Hub — Utility Functions."""

import re
import unicodedata
from typing import Optional

import pandas as pd
import numpy as np
from src.config import (DK_FINISH_POINTS, FD_FINISH_POINTS, DRIVER_ALIASES,
                        DK_PTS_LAP_LED, DK_PTS_FASTEST_LAP, DK_PTS_PLACE_DIFF)


def compute_practice_composite(
        lap_avg_df: pd.DataFrame,
        weights: dict = None) -> pd.DataFrame:
    """Composite practice scores from NASCAR's consecutive-lap-average windows.

    NASCAR doesn't publish lap-by-lap practice, but its lap-averages feed gives
    per driver: BestLapTime + best-N-consecutive-lap averages (5/10/15/20/25/30,
    999 = didn't run that long). That's enough to grade the same dimensions the
    public tools show — no secret data required:

      Peak        — top-end speed (best lap + 5-lap burst), field-relative.
      Consistency — how tightly the short run holds vs the single best lap
                    (small Con5−Best gap = repeatable).
      Fade        — tire falloff over a sustained run (longest window − Con5);
                    less falloff scores higher. Neutral when no long run exists.
      Shape       — sustained green-flag pace (the longest window's speed),
                    field-relative — "race pace", not one hot lap.
      Confidence  — how much usable run the driver actually put down (how many
                    windows are populated / longest run length).

    Composite = weighted blend (defaults mirror the common public weighting:
    peak .30 / consistency .22 / fade .18 / shape .18 / confidence .12).
    Each sub-score is 0–100 (higher = better). Returns a DataFrame with
    Driver, the five sub-scores, Composite, and a Profile Tag. Empty in →
    empty out.
    """
    W = {"peak": 0.30, "consistency": 0.22, "fade": 0.18,
         "shape": 0.18, "confidence": 0.12}
    if weights:
        W.update({k: weights[k] for k in W if k in weights})
    if lap_avg_df is None or lap_avg_df.empty or "Driver" not in lap_avg_df.columns:
        return pd.DataFrame()

    windows = [("5 Lap", 5), ("10 Lap", 10), ("15 Lap", 15),
               ("20 Lap", 20), ("25 Lap", 25), ("30 Lap", 30)]

    def _num(v):
        try:
            f = float(v)
            return f if 0 < f < 900 else None   # 999/None sentinel → missing
        except (TypeError, ValueError):
            return None

    recs = []
    for _, row in lap_avg_df.iterrows():
        best = _num(row.get("Best Lap"))
        con = {n: _num(row.get(col)) for col, n in windows}
        con5 = con.get(5)
        present = [(n, t) for n, t in con.items() if t is not None]
        longest_n, longest_t = (max(present, key=lambda x: x[0])
                                if present else (0, None))
        recs.append({
            "Driver": row["Driver"], "best": best, "con5": con5,
            "longest_n": longest_n, "longest_t": longest_t,
            "fade_raw": (longest_t - con5) if (longest_t is not None
                        and con5 is not None and longest_n >= 15) else None,
            "consist_raw": (con5 - best) if (con5 is not None
                           and best is not None) else None,
        })
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)

    def _pct(series, lower_better=True):
        """0–100 percentile score within the field (NaN-safe)."""
        s = pd.to_numeric(series, errors="coerce")
        valid = s.dropna()
        if valid.empty:
            return pd.Series([np.nan] * len(s), index=s.index)
        ranks = s.rank(pct=True, ascending=not lower_better)
        return (ranks * 100).round(1)

    df["Peak"] = _pct(df["best"], lower_better=True)
    df["Shape"] = _pct(df["longest_t"], lower_better=True)
    # Consistency: smaller best→5lap gap = better; absent → mid (50)
    df["Consistency"] = _pct(df["consist_raw"], lower_better=True).fillna(50.0)
    # Fade: smaller falloff = better; no long run → neutral 50 (can't judge)
    df["Fade"] = _pct(df["fade_raw"], lower_better=True).fillna(50.0)
    # Confidence: longest run length, scaled (30 laps = 100, 5 = ~30)
    df["Confidence"] = (df["longest_n"].clip(0, 30) / 30 * 100).round(1)

    df["Composite"] = (
        df["Peak"].fillna(40) * W["peak"]
        + df["Consistency"] * W["consistency"]
        + df["Fade"] * W["fade"]
        + df["Shape"].fillna(40) * W["shape"]
        + df["Confidence"] * W["confidence"]
    ).round(1)

    def _tag(r):
        peak, fade, cons = r["Peak"], r["Fade"], r["Consistency"]
        if pd.isna(peak):
            return "Limited Data"
        fast = peak >= 70
        volatile = (cons < 45) or (fade < 40)
        fades = fade < 45
        if fast and volatile:
            return "Fast but Volatile"
        if fast and fades:
            return "Fast with Some Fade"
        if fast:
            return "Fast & Clean"
        if peak < 35 and fade >= 60:
            return "Slow but Steady"
        if r["Confidence"] < 35:
            return "Short Sample"
        return "Balanced"

    df["Profile Tag"] = df.apply(_tag, axis=1)
    df["Run"] = df["longest_n"].map(lambda n: f"{int(n)}L" if n else "—")
    out = df[["Driver", "Run", "Composite", "Peak", "Consistency",
              "Fade", "Shape", "Confidence", "Profile Tag"]].copy()
    return out.sort_values("Composite", ascending=False).reset_index(drop=True)


# Per-lap-window base weights for the practice signal. Longer green-flag runs
# predict RACE pace far better than a single fast lap (which is qualifying-style
# speed), so they carry more weight. "Moderate" tilt: a fast 5-lap run still
# counts, but 10-30 lap runs dominate.
_PRACTICE_BUCKET_WEIGHTS = {
    "1 Lap Rank":  0.5,
    "5 Lap Rank":  0.8,
    "10 Lap Rank": 1.2,
    "15 Lap Rank": 1.5,
    "20 Lap Rank": 1.8,
    "25 Lap Rank": 2.0,
    "30 Lap Rank": 2.0,
}


def compute_practice_signals(lap_averages_df, field_size: int = None) -> dict:
    """Coverage-weighted practice signal for the whole field at once.

    Fixes two flaws in the old simple-mean-of-ranks approach:

    1. SPARSE-BUCKET INFLATION. NASCAR ranks each lap-window only among the
       drivers who actually ran that many consecutive laps. If just 3 cars ran
       a 30-lap run, their 30-lap ranks are 1/2/3 — small, elite-looking numbers
       — even though "3rd of 3" is dead LAST in that group. Averaging raw ranks
       rewarded drivers merely for running long. We fix this two ways:
         • Convert each rank to a PERCENTILE within its bucket:
               pct = (rank - 1) / (n_in_bucket - 1)
           so 1st-of-3 = 0.0 (best) and 3rd-of-3 = 1.0 (worst) — comparable to
           1st-of-38 = 0.0 and 38th-of-38 = 1.0.
         • Weight each bucket by COVERAGE = n_in_bucket / field_size, so a
           2-of-38 bucket contributes ~5% as much as a fully-run bucket. A
           handful of drivers running a long run can't manufacture value for
           themselves. Lone-runner buckets (n < 2) are dropped entirely.

    2. NO LONG-RUN EMPHASIS. Each window counted equally. Long runs predict
       race pace better, so they get higher BASE weight (_PRACTICE_BUCKET_WEIGHTS).

    Crucially this neither punishes a driver for running few laps nor rewards one
    for running many: a driver is judged purely on WHERE they placed in each
    bucket they ran, with sparsely-run buckets discounted for everyone.

    Args:
        lap_averages_df: DataFrame with a "Driver" column and any of the
            "{N} Lap Rank" columns (ranks are 1-based, lower = faster).
        field_size: denominator for coverage. Defaults to the row count.

    Returns:
        {driver_name: signal} where signal is in rank-like units (~1..field_size,
        lower = better) so it drops into the projection engine unchanged. Drivers
        with no usable buckets are omitted.
    """
    if lap_averages_df is None or lap_averages_df.empty:
        return {}
    df = lap_averages_df
    if "Driver" not in df.columns:
        return {}
    n_field = field_size or len(df)
    if n_field <= 1:
        n_field = max(len(df), 2)

    # Per-bucket participation counts, coverage, and base length-weight.
    bucket_info = {}  # col -> (n_in_bucket, coverage, base_weight)
    for col, base_w in _PRACTICE_BUCKET_WEIGHTS.items():
        if col not in df.columns:
            continue
        n_b = int(pd.to_numeric(df[col], errors="coerce").notna().sum())
        if n_b < 2:
            continue  # lone runner (or none) → no meaningful place to score
        coverage = n_b / n_field
        bucket_info[col] = (n_b, coverage, base_w)
    if not bucket_info:
        return {}

    out = {}
    for _, row in df.iterrows():
        driver = row.get("Driver")
        if not driver:
            continue
        wsum = 0.0
        wpct = 0.0
        for col, (n_b, coverage, base_w) in bucket_info.items():
            v = row.get(col)
            try:
                if v is None or pd.isna(v):
                    continue
                r = float(v)
            except (TypeError, ValueError):
                continue
            pct = (r - 1.0) / (n_b - 1.0)        # 0 = fastest in bucket, 1 = slowest
            pct = min(1.0, max(0.0, pct))
            # COVERAGE SHRINKAGE: a placement in a sparsely-run bucket is weak
            # evidence, so pull its value toward neutral (0.5) by how poorly the
            # bucket was covered. 1st-of-38 (coverage~1) stays elite (~0.0);
            # 1st-of-5 (coverage~0.13) lands ~0.44 (barely better than neutral);
            # 3rd-of-3 (~0.07) lands ~0.54 (near neutral, NOT "dead last"). This
            # fixes the single-sparse-bucket hole that weighting alone couldn't:
            # a driver whose ONLY run is 1st-of-3 no longer projects as the #1
            # practice car off a 3-car sample.
            pct_adj = pct * coverage + 0.5 * (1.0 - coverage)
            # Run-length emphasis (long runs predict race pace better) still
            # rides on top as the relative weight between a driver's buckets.
            eff_w = base_w * coverage
            wpct += pct_adj * eff_w
            wsum += eff_w
        if wsum <= 0:
            continue
        avg_pct = wpct / wsum
        # Back to rank-like units (lower = better) for the projection engine.
        out[driver] = 1.0 + avg_pct * (n_field - 1)
    return out


def parse_american_odds(value) -> Optional[int]:
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
        diff_pts = (int(start) - int(finish)) * DK_PTS_PLACE_DIFF
        led_pts = int(laps_led) * DK_PTS_LAP_LED
        fl_pts = int(fastest_laps) * DK_PTS_FASTEST_LAP
        return round(place_pts + diff_pts + led_pts + fl_pts, 2)
    except (ValueError, TypeError):
        return 0.0


def calc_fd_points(finish, start, laps_led, laps_completed=0):
    """Calculate FanDuel NASCAR DFS points.

    Official FD scoring (verified 2026-06): finish points (43/40/38, then -1
    per spot to 40th=1) + 0.5/position differential + 0.1/lap led +
    0.1/lap COMPLETED. FanDuel does NOT award fastest-lap points — the old
    version of this function wrongly paid 0.5/fastest lap and ignored laps
    completed.
    """
    try:
        place_pts = FD_FINISH_POINTS.get(int(finish), 0)
        diff_pts = (int(start) - int(finish)) * 0.5
        led_pts = int(laps_led) * 0.1
        comp_pts = int(laps_completed or 0) * 0.1
        return round(place_pts + diff_pts + led_pts + comp_pts, 2)
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


def _match_keys(name: str) -> list:
    """Return all normalized keys a name could match against.

    First key is the primary normalized form, plus a nickname-expanded
    variant (Nick -> Nicholas). We intentionally do NOT add a
    middle-stripped key: a middle initial present in one name but absent in
    the other indicates DIFFERENT drivers (Austin Hill vs Austin J Hill,
    Jason White vs Jason M White), not the same person. Middle-initial
    *abbreviation* of a present middle (John H ↔ John Hunter) is handled by
    the component-wise pass via _middles_compatible.
    """
    primary = normalize_driver_name(name)
    keys = [primary]
    # Nickname-expanded
    nc = _nickname_canonical(primary)
    if nc != primary:
        keys.append(nc)
    return keys


def _name_components(norm: str):
    """Split a normalized name into (first, [middle...], last).

    Suffixes are already stripped by normalize_driver_name. For a single
    token, returns it as the first name with empty middle/last.
    """
    parts = norm.split()
    if not parts:
        return "", [], ""
    if len(parts) == 1:
        return parts[0], [], ""
    return parts[0], parts[1:-1], parts[-1]


def _middles_compatible(a_mids: list, b_mids: list) -> bool:
    """True if two middle-token lists belong to the same driver.

    Rules (per user requirement):
      - both empty            -> same person, compatible
      - one empty, one present -> DIFFERENT people (e.g. Austin Hill vs
        Austin J Hill, Jason White vs Jason M White)
      - both present          -> compatible only if same count and each
        token pair is equal or an initial of the other (John H ↔ John Hunter)
    """
    if not a_mids and not b_mids:
        return True
    if bool(a_mids) != bool(b_mids):
        return False
    if len(a_mids) != len(b_mids):
        return False
    for a, b in zip(a_mids, b_mids):
        if a == b:
            continue
        if (len(a) == 1 and b.startswith(a)) or (len(b) == 1 and a.startswith(b)):
            continue
        return False
    return True


def _first_names_compatible(a: str, b: str) -> bool:
    """True if two first-name tokens plausibly belong to the same driver:
    identical, an initial of the other ("A" ↔ "Austin"), or nickname-equivalent
    (Nick ↔ Nicholas). Different real first names (Austin vs Timmy, Chandler vs
    Zane, Ed vs Erik) return False."""
    if not a or not b:
        return False
    if a == b:
        return True
    if (len(a) == 1 and b.startswith(a)) or (len(b) == 1 and a.startswith(b)):
        return True
    return _nickname_canonical(a) == _nickname_canonical(b)


def _surnames_compatible(a: str, b: str) -> bool:
    """True if two surname tokens plausibly belong to the same driver:
    identical, one a prefix of the other (truncated source), or high typo
    similarity (>=0.88). Different surnames (Butcher vs Custer) return False."""
    from difflib import SequenceMatcher
    if not a or not b:
        return False
    if a == b:
        return True
    if (len(a) >= 3 and b.startswith(a)) or (len(b) >= 3 and a.startswith(b)):
        return True
    # Typo tolerance. Safe at this level because the caller ALSO requires the
    # first names to be compatible, so same-surname/different-driver pairs
    # (Smith, Jones, Gilliland...) are already blocked by the first-name check.
    return SequenceMatcher(None, a, b).ratio() >= 0.85


def fuzzy_match_name(name: str, candidates: list, threshold: float = 0.75) -> str:
    """Find the match for a driver name in a list of candidates.

    Matching passes (first hit wins):
      1-3. Shared normalized key — exact (Suárez↔Suarez, periods/aliases),
           nickname (Nick↔Nicholas), or middle-initial (Jason M White↔Jason White).
      4.   Component-wise compatibility — BOTH the first name AND the surname
           must independently be compatible (see helpers). This is the only
           "fuzzy" path and it is deliberately strict: it will NOT merge two
           distinct drivers who happen to share one name component or have a
           high whole-string similarity. Examples it correctly REJECTS:
             Austin Hill ↔ Timmy Hill      (same surname, diff first)
             Cole Butcher ↔ Cole Custer    (same first, diff surname)
             Chandler Smith ↔ Zane Smith   (same surname, diff first)
             David Gilliland ↔ Todd Gilliland, Ed Jones ↔ Erik Jones
           Examples it correctly ACCEPTS:
             A Hill ↔ Austin Hill          (initial)
             Ryan Blany ↔ Ryan Blaney      (surname typo)
             Nick Sanchez ↔ Nicholas Sanchez (nickname)

    The `threshold` arg is retained for backward compatibility but the
    component-wise logic supersedes the old whole-string ratio cutoff.
    """
    from difflib import SequenceMatcher

    if not name or not candidates:
        return None

    name_keys = set(_match_keys(name))
    norm_candidates = [(c, _match_keys(c)) for c in candidates]

    # Pass 1-3: any shared normalized key = same driver
    for candidate, cand_keys in norm_candidates:
        if name_keys & set(cand_keys):
            return candidate

    # Pass 4: component-wise compatibility (first AND surname must both match).
    primary_norm = normalize_driver_name(name)
    name_first, name_mid, name_last = _name_components(primary_norm)
    if not name_last:
        return None  # single-token name — too ambiguous to fuzzy-match safely

    best_match, best_score = None, 0.0
    for candidate, cand_keys in norm_candidates:
        cand_first, cand_mid, cand_last = _name_components(cand_keys[0])
        if not cand_last:
            continue
        if (_first_names_compatible(name_first, cand_first)
                and _surnames_compatible(name_last, cand_last)
                and _middles_compatible(name_mid, cand_mid)):
            # Tie-break among compatible candidates by whole-string similarity
            score = SequenceMatcher(None, primary_norm, cand_keys[0]).ratio()
            if score > best_score:
                best_score = score
                best_match = candidate

    return best_match


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
    """Look up a driver in a dict using exact → normalized → fuzzy matching.

    Args:
        name: Driver name to look up.
        mapping: Original {display_name: value} dict.
        norm_cache: Optional pre-built {normalized_name: value} dict from
                    build_norm_lookup() — avoids rebuilding each call.

    Falls through to component-wise fuzzy_match_name() (the SAME matcher
    fuzzy_merge uses) when exact + normalized lookups miss — so first-name
    NICKNAMES the normalizer doesn't expand still resolve, e.g. "Nicholas
    Sanchez" ↔ the lap feed's "Nick Sanchez". fuzzy_match_name is middle-aware,
    so it still will NOT match "Justin S Carroll"→"Justin Carroll" or
    "Jason M White"→"Jason White" (middle initial on one side only → different
    people). Previously fuzzy_get stopped at the normalized step, so those
    nickname cases silently returned None.
    """
    if name in mapping:
        return mapping[name]
    norm = normalize_driver_name(name)
    if norm_cache is None:
        norm_cache = build_norm_lookup(mapping)
    if norm in norm_cache:
        return norm_cache[norm]
    matched = fuzzy_match_name(name, list(mapping.keys()))
    if matched is not None and matched in mapping:
        return mapping[matched]
    return None


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
