import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\codyr\OneDrive\Desktop\NASCAR DFS")
import pandas as pd, numpy as np

sp = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(sp, "audit_rows.csv"))

print("===== dom rank 1-6: what finish bucket do they project in? =====")
for rk in range(1, 7):
    g = df[df.ll_rank == rk]
    print(f"rank {rk}: mean proj finish {g.proj_finish.mean():5.1f} | median {g.proj_finish.median():5.1f} | "
          f"mean start {g.start.mean():4.1f}")

print("\n===== per-parent E[act LL share | model rank 1] vs allocated =====")
for parent in ["intermediate", "road", "short", "short_concrete", "superspeedway"]:
    g = df[(df.parent == parent) & (df.ll_rank == 1)]
    if len(g) == 0:
        continue
    print(f"{parent:>15}: n={len(g):>2} alloc {g.proj_ll.mean()/g.race_laps.mean():.3f} "
          f"| actual {g.act_ll.mean()/g.race_laps.mean():.3f}")

print("\n===== LL bias inside proj-finish 6-10 bucket, by dom rank =====")
b = df[(df.proj_finish >= 6) & (df.proj_finish < 10)]
for rk in range(1, 8):
    g = b[b.ll_rank == rk]
    if len(g) < 3:
        continue
    print(f"rank {rk}: n={len(g):>3} | proj LL {g.proj_ll.mean():5.1f} | act LL {g.act_ll.mean():5.1f} "
          f"| LL pts bias {(g.p_ll-g.a_ll).mean():+5.2f}")

print("\n===== who holds dom rank 1? by start pos =====")
r1 = df[df.ll_rank == 1]
print(r1.groupby(pd.cut(r1.start, [0, 3, 6, 10, 15, 40]), observed=True)[["proj_ll", "act_ll"]].agg(["count", "mean"]).round(1))

print("\n===== FL bias by START bucket and parent (P23+ under-projection source) =====")
for parent in ["intermediate", "superspeedway", "road"]:
    g = df[(df.parent == parent) & (df.start >= 23)]
    print(f"{parent:>15} P23+: n={len(g):>3} proj FL {g.proj_fl.mean():4.1f} act FL {g.act_fl.mean():4.1f} "
          f"pts bias {(g.p_fl-g.a_fl).mean():+5.2f}")

# ---- DNF cost estimate for the 10-15 bucket
print("\n===== DNF cost check (10-15 bucket) =====")
b = df[(df.proj_finish >= 10) & (df.proj_finish < 15)]
dnfs = b[b.dnf]
runs = b[~b.dnf]
print(f"bucket n={len(b)}, DNF n={len(dnfs)} ({len(dnfs)/len(b)*100:.1f}%)")
print(f"mean act DK: running {runs.a_dk.mean():.1f} vs DNF {dnfs.a_dk.mean():.1f} (delta {runs.a_dk.mean()-dnfs.a_dk.mean():.1f})")
print(f"unconditional bias explained by DNFs: {(len(dnfs)/len(b))*(runs.a_dk.mean()-dnfs.a_dk.mean()):.2f} pts")

# ---- Inspect calibration curves for representative tracks
from src.data import _get_track_dominator_calibration
print("\n===== CALIBRATION CURVES (live, no before_date) =====")
for track, tt, sid in [("Charlotte Motor Speedway", "intermediate", 1),
                       ("Kansas Speedway", "intermediate", 1),
                       ("Bristol Motor Speedway", "short_concrete", 1),
                       ("Martinsville Speedway", "short", 1),
                       ("Nashville Superspeedway", "intermediate", 3),
                       ("Talladega Superspeedway", "superspeedway", 1)]:
    cal = _get_track_dominator_calibration(track, tt, sid)
    ll = cal.get("ll_rank_distribution")
    fl = cal.get("fl_rank_distribution")
    print(f"\n{track} (series {sid}):")
    print(f"  avg_top_leader={cal.get('avg_top_leader'):.0f} max_ll={cal.get('max_laps_led')} "
          f"avg_fl_leader={cal.get('avg_fl_leader'):.0f} n_leaders={cal.get('avg_n_leaders'):.1f}")
    if ll:
        print(f"  LL curve top6: {[round(x,3) for x in ll[:6]]} (len {len(ll)})")
    if fl:
        print(f"  FL curve top6: {[round(x,3) for x in fl[:6]]} (len {len(fl)})")

# raw curve WITHOUT temperature smoothing, to size its effect
import sqlite3
from src.config import DB_PATH
conn = sqlite3.connect(str(DB_PATH))
def raw_curve(track, series_id, col):
    rows = conn.execute(f'''
        SELECT r.id, rr.{col} FROM race_results rr
        JOIN races r ON r.id = rr.race_id JOIN tracks t ON t.id = r.track_id
        WHERE t.name = ? AND r.series_id = ? AND COALESCE(r.is_exhibition,0)=0
          AND rr.{col} IS NOT NULL
        ORDER BY r.id, rr.{col} DESC''', (track, series_id)).fetchall()
    per_race = {}
    for rid, v in rows:
        per_race.setdefault(rid, []).append(v or 0)
    rank_fracs = {}
    n_races = 0
    for rid, vals in per_race.items():
        total = sum(vals)
        if total < 30:
            continue
        n_races += 1
        for rank, v in enumerate(vals):
            rank_fracs.setdefault(rank, []).append(v / total)
    curve = [float(np.mean(rank_fracs[r])) for r in sorted(rank_fracs)]
    s = sum(curve)
    return [c / s for c in curve], n_races

print("\n===== TEMPERATURE EFFECT (raw vs shipped curve) =====")
for track, sid in [("Charlotte Motor Speedway", 1), ("Bristol Motor Speedway", 1),
                   ("Martinsville Speedway", 1)]:
    rc, n = raw_curve(track, sid, "laps_led")
    T = max(0.75, min(1.0, n / 10.0))
    smoothed = [c ** T for c in rc]
    s = sum(smoothed)
    smoothed = [c / s for c in smoothed]
    print(f"{track}: n_races={n} T={T:.2f} raw top3={[round(x,3) for x in rc[:3]]} "
          f"-> smoothed top3={[round(x,3) for x in smoothed[:3]]}")
conn.close()
