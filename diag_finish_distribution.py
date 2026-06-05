"""Diagnosis: does projecting DK from an EXPECTED-finish DISTRIBUTION (rather than
a forced unique integer running order) reduce bias / improve accuracy?

Current engine (src/projections.py:708-711) ranks drivers on a continuous
`raw_finish` score, then assigns strict integers 1..N and computes DK from those
integers. That manufactures spread for clustered drivers (7 cars all ~P10 become
8,9,10,11,12,13,14) and converts ~1:1 into projected-DK spread.

This script replays every backtestable race and compares three ways to turn each
driver's CONTINUOUS raw_finish into projected DK, holding laps-led / fast-laps
allocations constant (they don't change):

  CURRENT   strict integer rank  -> DK              (today's behavior)
  POINT     DK( continuous raw_finish ) directly    (the under-spread extreme)
  EV(sig)   E[DK] over a Gaussian finish dist, per-driver independent
  EVn(sig)  E[DK] over a Sinkhorn-normalized dist (one car per position in expn)

Metrics (projected DK vs ACTUAL DK), pooled over all driver-races and split by
projected-finish bucket so the over/under-inflation pattern is visible:
  MAE, mean signed bias, Pearson r, Spearman rho.

Usage:  python diag_finish_distribution.py
"""
import sys, os, math, sqlite3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import DK_FINISH_POINTS
from src.projections import compute_projections
from backtest_weights import (backtestable_races, load_race, normalize_weights,
                              current_weights, spearman)

SIGMAS = [4, 6, 8, 10, 12]


def _finish_pts(pos):
    return DK_FINISH_POINTS.get(max(1, min(40, int(round(pos)))), 0)


def _gauss_weights(center, n, sigma):
    """Discrete Gaussian over positions 1..n centered at `center`, renormalized."""
    w = [math.exp(-0.5 * ((k - center) / sigma) ** 2) for k in range(1, n + 1)]
    s = sum(w)
    return [x / s for x in w] if s > 0 else [1.0 / n] * n


def _sinkhorn(mat, iters=15):
    """Normalize an (drivers x positions) probability matrix so every row sums to
    1 (each driver finishes somewhere) AND every column sums to 1 (each position
    taken by exactly one car, in expectation). Keeps the 'someone wins' mass."""
    m = [row[:] for row in mat]
    R, C = len(m), len(m[0])
    for _ in range(iters):
        for i in range(R):  # rows -> 1
            s = sum(m[i]) or 1.0
            for j in range(C):
                m[i][j] /= s
        for j in range(C):  # cols -> 1
            s = sum(m[i][j] for i in range(R)) or 1.0
            for i in range(R):
                m[i][j] /= s
    # final row-normalize so each driver's dist is a proper pmf
    for i in range(R):
        s = sum(m[i]) or 1.0
        for j in range(C):
            m[i][j] /= s
    return m


def project_detailed(race):
    """Run compute_projections and return per-driver rows incl. raw_finish."""
    wn = normalize_weights(current_weights(race["series_id"], race["parent"]),
                           has_odds=bool(race["odds_finish"]), has_prac=False)
    rows, _, _ = compute_projections(
        return_signal_details=True, drivers=race["drivers"],
        field_size=race["field_size"], wn=wn, th_data=race["th_data"],
        tt_data=race["tt_data"], qual_pos=race["qual_pos"], practice_data={},
        odds_finish=race["odds_finish"], odds_display=race["odds_display"],
        team_signal=race["team_signal"], mfr_adjustment={},
        team_adj_data=race["team_adj"], dnf_data={}, race_laps=200,
        track_name=race["track_name"], track_type=race["track_type"],
        series_id=race["series_id"], calibration=race["calibration"],
        cross_th_lookup={})
    return rows


def variants_for_race(race, rows):
    """Return {variant: {driver: proj_dk}} for one race."""
    n = race["field_size"]
    out = {"CURRENT": {}, "POINT": {}}
    for s in SIGMAS:
        out[f"EV{s}"] = {}
        out[f"EVn{s}"] = {}

    # Constant pieces per driver: start, laps/fl points; raw_finish center.
    info = {}
    centers, drivers = [], []
    for r in rows:
        d = r["driver"]
        info[d] = dict(start=r["start"], led=r["led_pts"], fl=r["fl_pts"],
                       raw=max(1.0, min(float(n), r["raw_finish"])),
                       proj_finish=r["proj_finish"], current=r["proj_dk"])
        out["CURRENT"][d] = r["proj_dk"]
        # POINT: DK straight off the continuous raw_finish (no distribution).
        c = info[d]["raw"]
        out["POINT"][d] = round(_finish_pts(c) + (r["start"] - c) + r["led_pts"] + r["fl_pts"], 2)
        centers.append(c); drivers.append(d)

    # Per-sigma EV (independent) and EVn (Sinkhorn-normalized).
    pos_pts = [_finish_pts(k) for k in range(1, n + 1)]
    for s in SIGMAS:
        mat = [_gauss_weights(c, n, s) for c in centers]
        for di, d in enumerate(drivers):
            w = mat[di]
            e_finish = sum((k + 1) * w[k] for k in range(n))
            e_pts = sum(pos_pts[k] * w[k] for k in range(n))
            out[f"EV{s}"][d] = round(e_pts + (info[d]["start"] - e_finish)
                                     + info[d]["led"] + info[d]["fl"], 2)
        matn = _sinkhorn(mat)
        for di, d in enumerate(drivers):
            w = matn[di]
            e_finish = sum((k + 1) * w[k] for k in range(n))
            e_pts = sum(pos_pts[k] * w[k] for k in range(n))
            out[f"EVn{s}"][d] = round(e_pts + (info[d]["start"] - e_finish)
                                      + info[d]["led"] + info[d]["fl"], 2)
    return out, info


def pearson(pairs):
    n = len(pairs)
    if n < 4:
        return float("nan")
    sx = sum(p for p, _ in pairs); sy = sum(a for _, a in pairs)
    mx, my = sx / n, sy / n
    cov = sum((p - mx) * (a - my) for p, a in pairs)
    vx = sum((p - mx) ** 2 for p, _ in pairs)
    vy = sum((a - my) ** 2 for _, a in pairs)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


def main():
    races = backtestable_races()
    conn = sqlite3.connect(str(__import__("src.config", fromlist=["DB_PATH"]).DB_PATH))

    variants = ["CURRENT", "POINT"] + [f"EV{s}" for s in SIGMAS] + [f"EVn{s}" for s in SIGMAS]
    # accumulators: variant -> list of (proj, actual); plus per-bucket bias
    pairs = {v: [] for v in variants}
    buckets = {"front P1-10": (1, 10), "mid P11-22": (11, 22), "back P23+": (23, 99)}
    bbias = {v: {b: [] for b in buckets} for v in variants}
    rho_per_race = {v: [] for v in variants}
    n_races = 0

    for rid, sid, season, rdate, track in races:
        race = load_race(conn, rid, sid, track, rdate)
        if not race:
            continue
        rows = project_detailed(race)
        if not rows:
            continue
        out, info = variants_for_race(race, rows)
        actual = race["actual_dk"]
        n_races += 1
        for v in variants:
            race_pairs = []
            for d, proj in out[v].items():
                if d not in actual:
                    continue
                a = actual[d]
                pairs[v].append((proj, a))
                race_pairs.append((proj, a))
                pf = info[d]["proj_finish"]
                for b, (lo, hi) in buckets.items():
                    if lo <= pf <= hi:
                        bbias[v][b].append(proj - a)
            r = spearman(race_pairs)
            if r is not None:
                rho_per_race[v].append(r)

    conn.close()

    def mae(v):
        return sum(abs(p - a) for p, a in pairs[v]) / len(pairs[v])

    def bias(v):
        return sum(p - a for p, a in pairs[v]) / len(pairs[v])

    print(f"\nReplayed {n_races} races, {len(pairs['CURRENT'])} driver-races.\n")
    print(f"{'variant':<10}{'MAE':>8}{'bias':>8}{'Pearson':>9}{'rho':>7}"
          f"{'  bias[front/mid/back]':>26}")
    print("-" * 72)
    for v in variants:
        rho = sum(rho_per_race[v]) / len(rho_per_race[v]) if rho_per_race[v] else float("nan")
        fb = sum(bbias[v]["front P1-10"]) / max(1, len(bbias[v]["front P1-10"]))
        mb = sum(bbias[v]["mid P11-22"]) / max(1, len(bbias[v]["mid P11-22"]))
        bb = sum(bbias[v]["back P23+"]) / max(1, len(bbias[v]["back P23+"]))
        print(f"{v:<10}{mae(v):>8.3f}{bias(v):>8.3f}{pearson(pairs[v]):>9.3f}"
              f"{rho:>7.3f}   {fb:>+6.2f}/{mb:>+6.2f}/{bb:>+6.2f}")
    print("\nLower MAE = better. bias near 0 = unbiased. Front/mid/back bias shows "
          "whether a method systematically over(+)/under(-) projects each tier.")


if __name__ == "__main__":
    main()
