"""Adversarial audit: decompose projected-vs-actual DK points by component
and by model dominator rank, across all backtestable races.

Read-only diagnostic. Mirrors scripts/calibration_study.py assembly.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\codyr\OneDrive\Desktop\NASCAR DFS")

import sqlite3
import numpy as np
from src.config import DB_PATH, TRACK_TYPE_WEIGHT_DEFAULTS, DK_FINISH_POINTS
from scripts.backtest_weights import backtestable_races, load_race, normalize_weights
from scripts.backtest_practice_weight import fetch_practice
from src.projections import compute_projections
from src.utils import normalize_driver_name, fuzzy_match_name

races = backtestable_races()
print(f"backtestable races: {len(races)}")

rows_all = []   # per driver-race dicts
race_level = [] # per race summaries
conn = sqlite3.connect(str(DB_PATH))
done = 0
for db_id, series_id, season, race_date, track in races:
    try:
        race = load_race(conn, db_id, series_id, track, race_date)
        if not race:
            continue
        api_row = conn.execute("SELECT api_race_id FROM races WHERE id=?",
                               (db_id,)).fetchone()
        api_id = api_row[0] if api_row else None
        prac = fetch_practice(season, series_id, api_id) or {} if api_id else {}
        if prac:
            norm = {normalize_driver_name(d): d for d in race["drivers"]}
            prac = {norm.get(normalize_driver_name(k),
                    fuzzy_match_name(k, race["drivers"]) or k): v
                    for k, v in prac.items()}
        base = TRACK_TYPE_WEIGHT_DEFAULTS.get(
            race["parent"], TRACK_TYPE_WEIGHT_DEFAULTS["intermediate"])
        wn = normalize_weights(base, has_odds=bool(race["odds_finish"]),
                               has_prac=bool(prac))
        proj_rows, detail = compute_projections(
            drivers=race["drivers"], field_size=race["field_size"], wn=wn,
            th_data=race["th_data"], tt_data=race["tt_data"],
            qual_pos=race["qual_pos"], practice_data=prac,
            odds_finish=race["odds_finish"], odds_display=race["odds_display"],
            team_signal=race["team_signal"], mfr_adjustment={},
            team_adj_data=race["team_adj"], dnf_data={},
            race_laps=race.get("race_laps", 200), track_name=track,
            track_type=race["track_type"], series_id=series_id,
            calibration=race["calibration"], cross_th_lookup={})
        race_laps = race.get("race_laps", 200)
        # actual components
        tot_proj_ll = sum(r["laps_led"] for r in proj_rows)
        tot_proj_fl = sum(r["fast_laps"] for r in proj_rows)
        tot_act_ll = sum(race["actual_ll"].values())
        tot_act_fl = sum(race["actual_fl"].values())
        # model dominator rank: by projected laps_led desc, then fl
        ll_sorted = sorted(proj_rows, key=lambda r: (r["laps_led"], r["fast_laps"]), reverse=True)
        ll_rank = {r["driver"]: i + 1 for i, r in enumerate(ll_sorted)}
        # actual LL rank
        act_ll_sorted = sorted(race["actual_ll"].items(), key=lambda kv: kv[1], reverse=True)
        act_ll_rank = {d: i + 1 for i, (d, v) in enumerate(act_ll_sorted)}
        act_leader = act_ll_sorted[0][0] if act_ll_sorted and act_ll_sorted[0][1] > 0 else None
        model_top = ll_sorted[0]["driver"] if ll_sorted else None
        cal = race["calibration"] or {}
        ll_dist = cal.get("ll_rank_distribution")
        fl_dist = cal.get("fl_rank_distribution")
        race_level.append(dict(
            track=track, series=series_id, date=race_date, laps=race_laps,
            parent=race["parent"],
            tot_proj_ll=tot_proj_ll, tot_act_ll=tot_act_ll,
            tot_proj_fl=tot_proj_fl, tot_act_fl=tot_act_fl,
            model_top_ll=ll_sorted[0]["laps_led"] if ll_sorted else 0,
            act_top_ll=act_ll_sorted[0][1] if act_ll_sorted else 0,
            model_top_is_leader=(model_top == act_leader),
            ll_dist_top=(ll_dist[0] if ll_dist else None),
            has_ll_dist=bool(ll_dist), has_fl_dist=bool(fl_dist),
            n_dist=len(ll_dist) if ll_dist else 0,
        ))
        for r in proj_rows:
            d = r["driver"]
            af = race["actual_finish"].get(d)
            if af is None:
                continue
            st = race["start_pos"][d]
            status = (race["status"].get(d) or "").lower()
            dnf = bool(status and "running" not in status and status.strip())
            act_fin_pts = DK_FINISH_POINTS.get(int(af), 0)
            act_diff = st - af
            act_ll_pts = (race["actual_ll"].get(d, 0)) * 0.25
            act_fl_pts = (race["actual_fl"].get(d, 0)) * 0.45
            rows_all.append(dict(
                race=track, date=race_date, series=series_id, parent=race["parent"],
                driver=d, start=st, dnf=dnf,
                proj_finish=r["proj_finish"], act_finish=af,
                p_fin=r["finish_pts"], p_diff=r["diff_pts"],
                p_ll=r["led_pts"], p_fl=r["fl_pts"], p_dk=r["proj_dk"],
                a_fin=act_fin_pts, a_diff=act_diff, a_ll=act_ll_pts, a_fl=act_fl_pts,
                a_dk=act_fin_pts + act_diff + act_ll_pts + act_fl_pts,
                ll_rank=ll_rank[d], act_ll=race["actual_ll"].get(d, 0),
                proj_ll=r["laps_led"], proj_fl=r["fast_laps"],
                act_fl=race["actual_fl"].get(d, 0),
                act_ll_rank=act_ll_rank.get(d, 99),
                race_laps=race_laps,
            ))
        done += 1
    except Exception as e:
        print(f"skip {track} {race_date}: {e}")
        continue
conn.close()
print(f"replayed: {done} races, {len(rows_all)} driver-races")

import pandas as pd
df = pd.DataFrame(rows_all)
rl = pd.DataFrame(race_level)

print("\n===== RACE-LEVEL TOTALS =====")
print("sum(proj LL) vs race_laps vs sum(actual LL):")
rl["proj_ll_frac"] = rl.tot_proj_ll / rl.laps
rl["act_ll_frac"] = rl.tot_act_ll / rl.laps
rl["proj_fl_frac"] = rl.tot_proj_fl / rl.laps
rl["act_fl_frac"] = rl.tot_act_fl / rl.laps
print(rl.groupby("parent")[["proj_ll_frac", "act_ll_frac", "proj_fl_frac", "act_fl_frac"]].mean().round(3))
print(f"\nmodel top-LL pick == actual LL leader: {rl.model_top_is_leader.mean()*100:.0f}% of races")
print(f"model top allocation (laps): mean {rl.model_top_ll.mean():.0f} vs actual top leader {rl.act_top_ll.mean():.0f}")
print(f"races with empirical ll_dist: {rl.has_ll_dist.mean()*100:.0f}%  fl_dist: {rl.has_fl_dist.mean()*100:.0f}%")
print("\nper-parent: model top-share vs actual top-share vs curve[0]")
rl["model_top_share"] = rl.model_top_ll / rl.laps
rl["act_top_share"] = rl.act_top_ll / rl.laps
print(rl.groupby("parent")[["model_top_share", "act_top_share", "ll_dist_top"]].mean().round(3))

print("\n===== BY MODEL DOMINATOR RANK (LL) =====")
print(f"{'rank':>4} | {'n':>4} | {'proj LL':>8} | {'act LL':>7} | {'bias pts':>8} | {'proj FL':>8} | {'act FL':>7}")
for rk in range(1, 13):
    m = df.ll_rank == rk
    if m.sum() == 0:
        continue
    print(f"{rk:>4} | {m.sum():>4} | {df[m].proj_ll.mean():>8.1f} | {df[m].act_ll.mean():>7.1f} | "
          f"{(df[m].p_ll - df[m].a_ll).mean():>+8.2f} | {df[m].proj_fl.mean():>8.1f} | {df[m].act_fl.mean():>7.1f}")

print("\n===== ACTUAL LL BY ACTUAL RANK (empirical concentration) =====")
for rk in range(1, 8):
    m = df.act_ll_rank == rk
    print(f"actual rank {rk}: mean laps {df[m].act_ll.mean():.1f} (share {df[m].act_ll.mean()/df[m].race_laps.mean():.3f})")

print("\n===== COMPONENT BIAS BY PROJECTED FINISH BUCKET =====")
print(f"{'bucket':>8} | {'n':>5} | {'fin':>6} | {'diff':>6} | {'LL':>6} | {'FL':>6} | {'total':>6}")
for lo, hi in [(1, 3), (3, 6), (6, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 43)]:
    m = (df.proj_finish >= lo) & (df.proj_finish < hi)
    if m.sum() < 30:
        continue
    b_fin = (df[m].p_fin - df[m].a_fin).mean()
    b_diff = (df[m].p_diff - df[m].a_diff).mean()
    b_ll = (df[m].p_ll - df[m].a_ll).mean()
    b_fl = (df[m].p_fl - df[m].a_fl).mean()
    b_tot = (df[m].p_dk - df[m].a_dk).mean()
    print(f"{f'{lo}-{hi}':>8} | {m.sum():>5} | {b_fin:>+6.2f} | {b_diff:>+6.2f} | {b_ll:>+6.2f} | {b_fl:>+6.2f} | {b_tot:>+6.2f}")

print("\nsame, RUNNING only (DNFs excluded):")
dfr = df[~df.dnf]
for lo, hi in [(1, 3), (3, 6), (6, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 43)]:
    m = (dfr.proj_finish >= lo) & (dfr.proj_finish < hi)
    if m.sum() < 30:
        continue
    b_fin = (dfr[m].p_fin - dfr[m].a_fin).mean()
    b_diff = (dfr[m].p_diff - dfr[m].a_diff).mean()
    b_ll = (dfr[m].p_ll - dfr[m].a_ll).mean()
    b_fl = (dfr[m].p_fl - dfr[m].a_fl).mean()
    b_tot = (dfr[m].p_dk - dfr[m].a_dk).mean()
    print(f"{f'{lo}-{hi}':>8} | {m.sum():>5} | {b_fin:>+6.2f} | {b_diff:>+6.2f} | {b_ll:>+6.2f} | {b_fl:>+6.2f} | {b_tot:>+6.2f}")

print("\n===== MAE overall =====")
print(f"DK points MAE: {(df.p_dk - df.a_dk).abs().mean():.2f}")
print(f"  finish comp MAE: {(df.p_fin - df.a_fin).abs().mean():.2f}")
print(f"  diff comp MAE:   {(df.p_diff - df.a_diff).abs().mean():.2f}")
print(f"  LL comp MAE:     {(df.p_ll - df.a_ll).abs().mean():.2f}")
print(f"  FL comp MAE:     {(df.p_fl - df.a_fl).abs().mean():.2f}")

# how many drivers get >=1 projected lap led / >= 5 laps?
print("\n===== SMEAR WIDTH =====")
per_race = df.groupby(["race", "date"]).apply(
    lambda g: pd.Series({
        "n_proj_ll_ge1": (g.proj_ll >= 1).sum(),
        "n_proj_ll_ge5": (g.proj_ll >= 5).sum(),
        "n_act_ll_ge1": (g.act_ll >= 1).sum(),
        "n_act_ll_ge5": (g.act_ll >= 5).sum(),
        "n_proj_fl_ge1": (g.proj_fl >= 1).sum(),
        "n_act_fl_ge1": (g.act_fl >= 1).sum(),
    }))
print(per_race.mean().round(1))

df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_rows.csv"), index=False)
print("\nsaved audit_rows.csv")

print("\n===== EXTRA: DNF rate + mean proj/act DK by projected-finish bucket =====")
for lo, hi in [(1, 3), (3, 6), (6, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 43)]:
    m = (df.proj_finish >= lo) & (df.proj_finish < hi)
    if m.sum() < 10:
        print(f"{lo}-{hi}: n={m.sum()} (too few)")
        continue
    print(f"{f'{lo}-{hi}':>8} | n={m.sum():>4} | proj DK {df[m].p_dk.mean():6.1f} | act DK {df[m].a_dk.mean():6.1f} | "
          f"DNF {df[m].dnf.mean()*100:4.1f}% | proj fin {df[m].proj_finish.mean():5.1f} | act fin {df[m].act_finish.mean():5.1f} | "
          f"act fin (run-only) {df[m & ~df.dnf].act_finish.mean():5.1f}")

print("\n===== EXTRA: model LL rank-1 outcome distribution =====")
r1 = df[df.ll_rank == 1]
print(f"n={len(r1)}; actual laps led by model's top pick: "
      f"p10={r1.act_ll.quantile(.1):.0f} p25={r1.act_ll.quantile(.25):.0f} "
      f"median={r1.act_ll.median():.0f} p75={r1.act_ll.quantile(.75):.0f} mean={r1.act_ll.mean():.0f}")
print(f"top pick led <10 laps in {(r1.act_ll < 10).mean()*100:.0f}% of races; "
      f"<25 laps in {(r1.act_ll < 25).mean()*100:.0f}%")
r2 = df[df.ll_rank == 2]
print(f"rank-2 pick led <10 laps in {(r2.act_ll < 10).mean()*100:.0f}%")

print("\n===== EXTRA: start-gate sanity (deep starters) =====")
for lo, hi, lbl in [(1, 5, "P1-5"), (6, 10, "P6-10"), (11, 15, "P11-15"),
                    (16, 22, "P16-22"), (23, 43, "P23+")]:
    m = (df.start >= lo) & (df.start <= hi)
    print(f"{lbl:>7} | n={m.sum():>4} | proj LL {df[m].proj_ll.mean():6.1f} | act LL {df[m].act_ll.mean():6.1f} | "
          f"proj FL {df[m].proj_fl.mean():5.1f} | act FL {df[m].act_fl.mean():5.1f}")

print("\n===== EXTRA: per-parent counts =====")
print(rl.groupby("parent").size())

print("\n===== EXTRA: E[LL share | model rank] predictive curve vs empirical curve =====")
df["act_share"] = df.act_ll / df.race_laps
df["proj_share"] = df.proj_ll / df.race_laps
pred_curve = df.groupby("ll_rank")[["proj_share", "act_share"]].mean().head(14)
print(pred_curve.round(3))
