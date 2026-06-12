"""Grid-search weight allocations per track type, with overfitting guards.

Searches a constrained neighborhood of allocations (5-pt steps, sum=100) for
each parent track type that has enough backtestable races, grading every
combo two ways (clean = running finishers only, primary; DNF-included =
sanity) plus a SPLIT-HALF stability check: races are split odd/even by date
and a winning combo must beat the current defaults on BOTH halves — with
n~12 races a full grid WILL find lucky combos, and the split-half agreement
is what separates signal from luck.

Usage: python scripts/backtest_grid_search.py
"""
import sys, os, sqlite3, itertools, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TRACK_TYPE_WEIGHT_DEFAULTS, DB_PATH
from scripts.backtest_weights import load_race, spearman
from scripts.backtest_practice_weight import (backtestable_races_with_api,
                                              fetch_practice, project)
from src.utils import normalize_driver_name, fuzzy_match_name

# Search menu per signal: tight 5-pt-step neighborhoods. Wider menus explode
# the combo count AND the overfitting risk; this is a refinement search, not
# a from-scratch one.
MENU = {
    "odds":  [15, 20, 25, 30],
    "track": [15, 20, 25, 30],
    "ttype": [5, 10, 15, 20],
    "prac":  [5, 10, 15, 20],
    "team":  [5, 10, 15],
    "qual":  [10, 15, 20, 25],
}
MIN_RACES = 8   # don't even attempt per-type tuning below this


def combos():
    keys = list(MENU)
    out = []
    for vals in itertools.product(*(MENU[k] for k in keys)):
        if sum(vals) == 100:
            out.append(dict(zip(keys, vals)))
    return out


def grade(race, prac, raw):
    """(rho_included, rho_clean) for one race under one allocation."""
    proj = project(race, raw, prac)
    status = race.get("status", {})

    def _running(d):
        s = status.get(d, "").lower()
        return s in ("running", "finished", "")

    overall = [(proj[d], race["actual_dk"][d]) for d in race["drivers"] if d in proj]
    clean = [(proj[d], race["actual_dk"][d]) for d in race["drivers"]
             if d in proj and _running(d)]
    return spearman(overall), spearman(clean)


def main():
    races_meta = backtestable_races_with_api()
    conn = sqlite3.connect(str(DB_PATH))
    loaded = []   # (race, prac, parent)
    for rid, sid, season, rdate, track, api_id in races_meta:
        race = load_race(conn, rid, sid, track, rdate)
        if not race:
            continue
        prac = fetch_practice(season, sid, api_id)
        if prac:
            norm = {normalize_driver_name(d): d for d in race["drivers"]}
            remapped = {}
            for k, v in prac.items():
                if k in race["drivers"]:
                    remapped[k] = v; continue
                nk = normalize_driver_name(k)
                if nk in norm:
                    remapped[norm[nk]] = v
                else:
                    m = fuzzy_match_name(k, race["drivers"])
                    if m:
                        remapped[m] = v
            prac = remapped
        loaded.append((race, prac, race["parent"], rdate))
    conn.close()

    all_combos = combos()
    print(f"races loaded: {len(loaded)} | combos per type: {len(all_combos)}")

    by_parent = {}
    for item in loaded:
        by_parent.setdefault(item[2], []).append(item)

    for parent, items in sorted(by_parent.items()):
        n = len(items)
        if n < MIN_RACES:
            print(f"\n### {parent}: only {n} races — too few to tune, skipping "
                  f"(needs {MIN_RACES}+; grows weekly as odds are saved)")
            continue
        items = sorted(items, key=lambda x: x[3])           # by date
        half_a = items[0::2]
        half_b = items[1::2]
        current = dict(TRACK_TYPE_WEIGHT_DEFAULTS[parent])

        t0 = time.time()
        scored = []
        for combo in all_combos:
            inc, cln, ca, cb = [], [], [], []
            for idx, (race, prac, _, _) in enumerate(items):
                ri, rc = grade(race, prac, combo)
                if ri is not None:
                    inc.append(ri)
                if rc is not None:
                    cln.append(rc)
                    (ca if idx % 2 == 0 else cb).append(rc)
            if not cln:
                continue
            scored.append({
                "combo": combo,
                "clean": sum(cln) / len(cln),
                "included": sum(inc) / len(inc) if inc else float("nan"),
                "half_a": sum(ca) / len(ca) if ca else float("nan"),
                "half_b": sum(cb) / len(cb) if cb else float("nan"),
            })
        elapsed = time.time() - t0

        # Current defaults' scores
        cur = next((s for s in scored
                    if s["combo"] == current), None)
        if cur is None:
            inc, cln, ca, cb = [], [], [], []
            for idx, (race, prac, _, _) in enumerate(items):
                ri, rc = grade(race, prac, current)
                if ri is not None:
                    inc.append(ri)
                if rc is not None:
                    cln.append(rc)
                    (ca if idx % 2 == 0 else cb).append(rc)
            cur = {"combo": current, "clean": sum(cln)/len(cln),
                   "included": sum(inc)/len(inc),
                   "half_a": sum(ca)/len(ca) if ca else float("nan"),
                   "half_b": sum(cb)/len(cb) if cb else float("nan")}

        scored.sort(key=lambda s: s["clean"], reverse=True)
        rank_cur = next((i for i, s in enumerate(scored)
                         if s["combo"] == current), None)

        def fmt(s, tag=""):
            c = s["combo"]
            return (f"  {tag:<9}odds{c['odds']:>3} trk{c['track']:>3} "
                    f"tt{c['ttype']:>3} prac{c['prac']:>3} team{c['team']:>3} "
                    f"qual{c['qual']:>3} | clean {s['clean']:.4f} "
                    f"incl {s['included']:.4f} | halves {s['half_a']:.3f}/{s['half_b']:.3f}")

        print(f"\n### {parent} — {n} races, {len(scored)} combos, {elapsed:.0f}s")
        print(fmt(cur, "CURRENT") + (f"  (rank {rank_cur + 1}/{len(scored)})"
                                     if rank_cur is not None else ""))
        print("  top 8 by clean rho:")
        for s in scored[:8]:
            # Robustness flags: must beat current on BOTH halves and on the
            # DNF-included sanity grading to be considered real.
            robust = (s["half_a"] > cur["half_a"] and s["half_b"] > cur["half_b"]
                      and s["included"] > cur["included"])
            print(fmt(s) + ("   <-- ROBUST" if robust else ""))


if __name__ == "__main__":
    main()
