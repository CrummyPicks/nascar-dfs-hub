"""Targeted backtest: should PRACTICE carry more weight (at track type's expense)?

Extends backtest_weights.py by fetching each historical race's practice
lap-averages from NASCAR's CDN (the files persist for completed races) and
computing the same coverage-weighted practice signal the live app uses.
Races without usable practice data still run, but with practice weight
renormalized away — identical across schemes, so they don't bias the
comparison. The races WITH practice are where the schemes differ; we report
that subset separately.

Usage: python scripts/backtest_practice_weight.py
"""
import sys, os, sqlite3, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd

from src.config import DB_PATH, TRACK_TYPE_WEIGHT_DEFAULTS, NASCAR_API_BASE
from src.utils import compute_practice_signals
from scripts.backtest_weights import (load_race, normalize_weights, spearman,
                                      SERIES_NAME)
from src.projections import compute_projections

# Reuse the lap-averages parser without Streamlit caching
from src.data import _parse_lap_avg_session
_parse = getattr(_parse_lap_avg_session, "__wrapped__", _parse_lap_avg_session)

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


def backtestable_races_with_api():
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute('''
        SELECT r.id, r.series_id, r.season, r.race_date, t.name, r.api_race_id
        FROM races r JOIN tracks t ON t.id = r.track_id
        WHERE r.api_race_id IS NOT NULL
          AND (SELECT COUNT(*) FROM odds o WHERE o.race_id = r.id) > 0
          AND (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.id) > 0
        ORDER BY r.series_id, r.race_date
    ''').fetchall()
    conn.close()
    return rows


def fetch_practice(season, series_id, api_race_id):
    """Practice signal dict for a historical race ({} if no usable data).
    Disk-cached so reruns are instant."""
    cpath = os.path.join(_CACHE_DIR, f"lapavg_{season}_{series_id}_{api_race_id}.json")
    data = None
    if os.path.exists(cpath):
        try:
            with open(cpath) as f:
                data = json.load(f)
        except Exception:
            data = None
    if data is None:
        try:
            r = requests.get(
                f"{NASCAR_API_BASE}/{season}/{series_id}/{api_race_id}/lap-averages.json",
                timeout=15)
            data = r.json() if r.status_code == 200 else []
        except Exception:
            data = []
        try:
            with open(cpath, "w") as f:
                json.dump(data, f)
        except Exception:
            pass
    if not data or not isinstance(data, list):
        return {}
    # UNION all practice session blocks — when NASCAR splits practice into
    # groups, each driver appears in exactly one block, so parsing only the
    # last block dropped everyone who ran in an earlier group (the same bug
    # fixed in the live fetch_lap_averages; mirrors its union + re-rank).
    import pandas as _pd
    frames = [f for f in (_parse(s) for s in data) if not f.empty]
    if not frames:
        return {}
    df = _pd.concat(frames, ignore_index=True)
    if "Driver" in df.columns:
        df = df.drop_duplicates("Driver", keep="first").reset_index(drop=True)
    for time_col, rank_col in [("Overall Avg", "Overall Rank"), ("Best Lap", "1 Lap Rank"),
                               ("5 Lap", "5 Lap Rank"), ("10 Lap", "10 Lap Rank"),
                               ("15 Lap", "15 Lap Rank"), ("20 Lap", "20 Lap Rank"),
                               ("25 Lap", "25 Lap Rank"), ("30 Lap", "30 Lap Rank")]:
        if time_col in df.columns and rank_col in df.columns:
            df[rank_col] = _pd.to_numeric(
                df[time_col], errors="coerce").rank(method="min")
    if df.empty:
        return {}
    has_ranks = any(c in df.columns for c in
                    ["1 Lap Rank", "5 Lap Rank", "10 Lap Rank", "15 Lap Rank",
                     "20 Lap Rank", "25 Lap Rank", "30 Lap Rank"])
    if not has_ranks:
        return {}
    return compute_practice_signals(df, field_size=len(df)) or {}


def project(race, raw_weights, practice_data):
    has_prac = bool(practice_data)
    wn = normalize_weights(raw_weights, has_odds=bool(race["odds_finish"]),
                           has_prac=has_prac)
    rows, _, _ = compute_projections(
        return_signal_details=True, drivers=race["drivers"], field_size=race["field_size"],
        wn=wn, th_data=race["th_data"], tt_data=race["tt_data"], qual_pos=race["qual_pos"],
        practice_data=practice_data or {}, odds_finish=race["odds_finish"],
        odds_display=race["odds_display"], team_signal=race["team_signal"],
        mfr_adjustment={}, team_adj_data=race["team_adj"], dnf_data={},
        race_laps=race.get("race_laps", 200), track_name=race["track_name"], track_type=race["track_type"],
        series_id=race["series_id"], calibration=race["calibration"], cross_th_lookup={})
    return {r["driver"]: r["proj_dk"] for r in rows}


def shifted(base, d_prac, d_ttype):
    out = dict(base)
    out["prac"] = max(0, out["prac"] + d_prac)
    out["ttype"] = max(0, out["ttype"] + d_ttype)
    return out


def main():
    schemes = {
        "A: current defaults":      lambda b: dict(b),
        "B: prac+5 / ttype-5":      lambda b: shifted(b, +5, -5),
        "C: prac+10 / ttype-10":    lambda b: shifted(b, +10, -10),
        "D: prac+5 / qual-5":       lambda b: {**dict(b), "prac": b["prac"] + 5,
                                               "qual": max(0, b["qual"] - 5)},
        # E: the per-type data says practice's edge lives at intermediates and
        # roads; short/concrete prefer the current mix. Shift only there.
        "E: B at int+road only":    None,  # handled specially below
        # Z: reconstruct the PRE-2026-06-11 defaults (prac -5 / ttype +5 at
        # int+road) so old-vs-shipped can be compared under both gradings.
        "Z: old defaults":          "revert",
    }
    races = backtestable_races_with_api()
    print(f"backtestable races: {len(races)}")
    conn = sqlite3.connect(str(DB_PATH))

    # rows: per scheme -> list of (rho_overall, rho_deep, had_practice, parent)
    results = {k: [] for k in schemes}
    n_prac = 0
    for rid, sid, season, rdate, track, api_id in races:
        race = load_race(conn, rid, sid, track, rdate)
        if not race:
            continue
        prac = fetch_practice(season, sid, api_id)
        # Match practice keys to the result-driver spellings
        if prac:
            from src.utils import normalize_driver_name, fuzzy_match_name
            norm = {normalize_driver_name(d): d for d in race["drivers"]}
            remapped = {}
            for k, v in prac.items():
                if k in race["drivers"]:
                    remapped[k] = v
                    continue
                nk = normalize_driver_name(k)
                if nk in norm:
                    remapped[norm[nk]] = v
                else:
                    m = fuzzy_match_name(k, race["drivers"])
                    if m:
                        remapped[m] = v
            prac = remapped
        if prac:
            n_prac += 1
        status = race.get("status", {})

        def _running(d):
            s = status.get(d, "").lower()
            return s in ("running", "finished", "")

        for label, fn in schemes.items():
            base = TRACK_TYPE_WEIGHT_DEFAULTS.get(
                race["parent"], TRACK_TYPE_WEIGHT_DEFAULTS["intermediate"])
            if fn is None:  # scheme E: shift only at intermediate/road
                raw = (shifted(base, +5, -5)
                       if race["parent"] in ("intermediate", "road") else dict(base))
            elif fn == "revert":  # scheme Z: undo the shipped int/road shift
                raw = (shifted(base, -5, +5)
                       if race["parent"] in ("intermediate", "road") else dict(base))
            else:
                raw = fn(base)
            proj = project(race, raw, prac)
            overall = [(proj[d], race["actual_dk"][d]) for d in race["drivers"] if d in proj]
            deep = [(proj[d], race["actual_dk"][d]) for d in race["drivers"]
                    if d in proj and race["start_pos"][d] >= 20]
            # Clean grading: running finishers only — measures pace-prediction
            # quality without wreck noise (a lap-3 dump scores ~5 DK no matter
            # how good the projection was).
            clean = [(proj[d], race["actual_dk"][d]) for d in race["drivers"]
                     if d in proj and _running(d)]
            results[label].append((spearman(overall), spearman(deep),
                                   bool(prac), race["parent"], spearman(clean)))
    conn.close()

    print(f"races with usable practice data: {n_prac}")
    print()
    print(f"{'scheme':<26}{'n':>5}{'rho_all':>9}{'rho_deep':>10}{'rho_clean':>10}"
          f"{'  | prac races':>14}{'rho_all':>9}{'rho_clean':>10}")
    for label, rows in results.items():
        f = lambda v: sum(v)/len(v) if v else float('nan')
        ao = [r[0] for r in rows if r[0] is not None]
        ad = [r[1] for r in rows if r[1] is not None]
        ac = [r[4] for r in rows if r[4] is not None]
        po = [r[0] for r in rows if r[0] is not None and r[2]]
        pc = [r[4] for r in rows if r[4] is not None and r[2]]
        print(f"{label:<26}{len(ao):>5}{f(ao):>9.4f}{f(ad):>10.4f}{f(ac):>10.4f}"
              f"{len(po):>14}{f(po):>9.4f}{f(pc):>10.4f}")

    # Per-track-type breakdown on practice races only (where schemes differ)
    print()
    print("Practice-races only, by track type (rho_all):")
    parents = sorted(set(r[3] for r in results["A: current defaults"] if r[2]))
    hdr = f"{'parent type':<18}" + "".join(f"{k.split(':')[0]:>8}" for k in schemes)
    print(hdr + f"{'races':>7}")
    for p in parents:
        line = f"{p:<18}"
        n = 0
        for label in schemes:
            vals = [r[0] for r in results[label]
                    if r[0] is not None and r[2] and r[3] == p]
            n = len(vals)
            line += f"{(sum(vals)/len(vals) if vals else float('nan')):>8.3f}"
        print(line + f"{n:>7}")


if __name__ == "__main__":
    main()
