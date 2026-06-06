"""Experiment: heteroscedastic (driver-strength-dependent) finish spread.

Flat sigma applies the same finishing variance to a dominant favorite and a
midpack car. This tests a 'tent' sigma: tight at the front (favorites convert),
widest in the midpack (real coin-flip), tighter again at the back (slow cars are
reliably slow). Goal: favorites project top-3-5 with ~neutral PD WITHOUT wrecking
the calibration that flat sigma=10 achieved.

Reports MAE / bias / per-tier bias (vs actual DK) AND the avg projected finish of
each race's strongest and weakest car, for flat-10 vs a few tent settings.
"""
import sys, os, math, sqlite3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import DK_FINISH_POINTS
from backtest_weights import backtestable_races, load_race
from diag_finish_distribution import project_detailed, pearson
from backtest_weights import spearman


def _finish_pts(pos):
    return DK_FINISH_POINTS.get(max(1, min(40, int(round(pos)))), 0)


def _sigma_flat(r, n, smin, smax):
    return smax


def _sigma_tent(r, n, smin, smax):
    if n <= 1:
        return smax
    frac = (min(max(r, 1.0), n) - 1) / (n - 1)      # 0=front .. 1=back
    tent = 1 - abs(2 * frac - 1)                      # 0 at extremes, 1 mid
    return smin + (smax - smin) * tent


def _sigma_ramp(r, n, smin, smax):
    """Asymmetric: tight at the FRONT only, ramping to full width by the midpack
    and STAYING wide through the back (backmarkers keep their attrition upside)."""
    if n <= 1:
        return smax
    frac = (min(max(r, 1.0), n) - 1) / (n - 1)      # 0=front .. 1=back
    return smin + (smax - smin) * min(1.0, frac / 0.4)   # full width by ~40% deep


def expectations(raw, drivers, n, sigfn, smin, smax, iters=20):
    order = list(drivers)
    if n <= 1:
        return {d: (1.0, _finish_pts(1)) for d in order}
    mat = []
    for d in order:
        c = max(1.0, min(float(n), raw.get(d, n * 0.75)))
        s = sigfn(c, n, smin, smax)
        row = [math.exp(-0.5 * ((k - c) / s) ** 2) for k in range(1, n + 1)]
        z = sum(row) or 1.0
        mat.append([x / z for x in row])
    R = len(mat)
    for _ in range(iters):
        for i in range(R):
            z = sum(mat[i]) or 1.0
            mat[i] = [x / z for x in mat[i]]
        for j in range(n):
            cs = sum(mat[i][j] for i in range(R)) or 1.0
            for i in range(R):
                mat[i][j] /= cs
    pts = [_finish_pts(k) for k in range(1, n + 1)]
    out = {}
    for i, d in enumerate(order):
        z = sum(mat[i]) or 1.0
        w = [x / z for x in mat[i]]
        out[d] = (sum((k + 1) * w[k] for k in range(n)),
                  sum(pts[k] * w[k] for k in range(n)))
    return out


CONFIGS = [
    ("flat-10",    _sigma_flat, 10, 10),
    ("tent 4-11",  _sigma_tent,  4, 11),
    ("ramp 4-11",  _sigma_ramp,  4, 11),
    ("ramp 3-11",  _sigma_ramp,  3, 11),
    ("ramp 4-12",  _sigma_ramp,  4, 12),
]


def main():
    races = backtestable_races()
    conn = sqlite3.connect(str(__import__("src.config", fromlist=["DB_PATH"]).DB_PATH))
    agg = {name: {"pairs": [], "fb": [], "mb": [], "bb": [], "fav": [], "last": [],
                  "favpd": []} for name, *_ in CONFIGS}
    for rid, sid, season, rdate, track in races:
        race = load_race(conn, rid, sid, track, rdate)
        if not race:
            continue
        rows = project_detailed(race)
        if not rows:
            continue
        n = race["field_size"]
        raw = {r["driver"]: r["raw_finish"] for r in rows}
        info = {r["driver"]: r for r in rows}
        actual = race["actual_dk"]
        fav = min(rows, key=lambda r: r["raw_finish"])["driver"]   # strongest
        last = max(rows, key=lambda r: r["raw_finish"])["driver"]  # weakest
        for name, fn, smin, smax in CONFIGS:
            exp = expectations(raw, race["drivers"], n, fn, smin, smax)
            A = agg[name]
            for d, (ef, ep) in exp.items():
                if d not in actual:
                    continue
                proj = ep + (info[d]["start"] - ef) + info[d]["led_pts"] + info[d]["fl_pts"]
                A["pairs"].append((proj, actual[d]))
                pf = ef
                (A["fb"] if pf <= 10 else A["mb"] if pf <= 22 else A["bb"]).append(proj - actual[d])
            A["fav"].append(exp[fav][0])
            A["last"].append(exp[last][0])
            A["favpd"].append(info[fav]["start"] - exp[fav][0])
    conn.close()

    def mae(p): return sum(abs(a - b) for a, b in p) / len(p)
    def bias(p): return sum(a - b for a, b in p) / len(p)
    def avg(x): return sum(x) / len(x) if x else float("nan")
    print(f"\n{'config':<11}{'MAE':>7}{'bias':>7}{'Pear':>7}"
          f"{'  f/m/b bias':>20}{'  favFin':>9}{'favPD':>7}{'lastFin':>8}")
    print("-" * 78)
    for name, *_ in CONFIGS:
        A = agg[name]
        print(f"{name:<11}{mae(A['pairs']):>7.2f}{bias(A['pairs']):>7.2f}"
              f"{pearson(A['pairs']):>7.3f}"
              f"   {avg(A['fb']):>+5.1f}/{avg(A['mb']):>+5.1f}/{avg(A['bb']):>+5.1f}"
              f"{avg(A['fav']):>9.1f}{avg(A['favpd']):>7.1f}{avg(A['last']):>8.1f}")
    print("\nfavFin = avg projected finish of each race's STRONGEST car; "
          "favPD = its avg place differential; lastFin = weakest car's finish.")


if __name__ == "__main__":
    main()
