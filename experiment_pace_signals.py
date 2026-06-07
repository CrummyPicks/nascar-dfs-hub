"""Do LONG-RUN / RESTART pace add predictive value beyond ARP?

Leak-free: for each Cup race, use each driver's PRIOR-race history to predict his
finish in THIS race.
    prior_arp  = recency-weighted avg running position over prior races (pace)
    prior_lr   = avg long-run rank over prior races (sustained green speed)
    prior_rs   = avg restart rank over prior races (short-run speed)
    fin        = actual finish
Then partial corr(prior_lr, fin | prior_arp) and (prior_rs, fin | prior_arp) —
the INDEPENDENT value beyond pace. ~0 => redundant with ARP (no projection gain);
clearly >0 => worth adding as a signal.
"""
import sys, os, math, sqlite3, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import DB_PATH

MIN_PRIOR = 3
MIN_PACE = 3   # need this many prior races WITH run_pace for a pace history


def pearson(xs, ys):
    n = len(xs)
    if n < 8:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs); vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


def partial(xs, ys, zs):
    rxy, rxz, ryz = pearson(xs, ys), pearson(xs, zs), pearson(ys, zs)
    den = math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / den if den and not math.isnan(den) else float("nan")


def rweight(vals):
    if not vals:
        return None
    w = [0.97 ** k for k in range(len(vals))][::-1]
    return sum(v * wt for v, wt in zip(vals, w)) / sum(w)


def main():
    conn = sqlite3.connect(str(DB_PATH))
    rr = conn.execute('''
        SELECT rr.driver_id, rr.finish_pos, rr.avg_running_position, r.race_date
        FROM race_results rr JOIN races r ON r.id = rr.race_id
        WHERE r.series_id = 1 AND r.season >= 2022 AND rr.finish_pos IS NOT NULL
    ''').fetchall()
    rp = conn.execute('''
        SELECT rp.driver_id, rp.long_run_rank, rp.restart_rank, r.race_date
        FROM run_pace rp JOIN races r ON r.id = rp.race_id
        WHERE r.series_id = 1 AND r.season >= 2022
    ''').fetchall()
    conn.close()

    arp_hist, lr_hist, rs_hist = {}, {}, {}
    for did, fin, arp, date in rr:
        if arp is not None:
            arp_hist.setdefault(did, []).append((date, arp))
    for did, lr, rs, date in rp:
        if lr is not None:
            lr_hist.setdefault(did, []).append((date, lr))
        if rs is not None:
            rs_hist.setdefault(did, []).append((date, rs))
    for h in (arp_hist, lr_hist, rs_hist):
        for v in h.values():
            v.sort()

    def prior(hist, did, date, minn):
        h = hist.get(did)
        if not h:
            return None
        i = bisect.bisect_left(h, (date,))
        vals = [v for _, v in h[:i]]
        return rweight(vals) if len(vals) >= minn else None

    lr_x, lr_y, lr_z = [], [], []   # long-run, finish, arp
    rs_x, rs_y, rs_z = [], [], []
    for did, fin, arp, date in rr:
        pa = prior(arp_hist, did, date, MIN_PRIOR)
        if pa is None:
            continue
        pl = prior(lr_hist, did, date, MIN_PACE)
        if pl is not None:
            lr_x.append(pl); lr_y.append(float(fin)); lr_z.append(pa)
        pr = prior(rs_hist, did, date, MIN_PACE)
        if pr is not None:
            rs_x.append(pr); rs_y.append(float(fin)); rs_z.append(pa)

    print(f"\nLong-run sample: {len(lr_x)} driver-races | Restart sample: {len(rs_x)}\n")
    print("LONG-RUN pace (prior avg rank):")
    print(f"  corr(prior_LR, finish)            = {pearson(lr_x, lr_y):+.3f}")
    print(f"  corr(prior_ARP, finish)           = {pearson(lr_z, lr_y):+.3f}  (benchmark)")
    print(f"  corr(prior_LR, prior_ARP)         = {pearson(lr_x, lr_z):+.3f}  (redundancy)")
    print(f"  PARTIAL(LR, finish | ARP)         = {partial(lr_x, lr_y, lr_z):+.3f}  <-- independent value")
    print("\nRESTART pace (prior avg rank):")
    print(f"  corr(prior_RS, finish)            = {pearson(rs_x, rs_y):+.3f}")
    print(f"  corr(prior_ARP, finish)           = {pearson(rs_z, rs_y):+.3f}  (benchmark)")
    print(f"  corr(prior_RS, prior_ARP)         = {pearson(rs_x, rs_z):+.3f}  (redundancy)")
    print(f"  PARTIAL(RS, finish | ARP)         = {partial(rs_x, rs_y, rs_z):+.3f}  <-- independent value")
    print("\n>0.05ish partial => worth adding as a signal; ~0 => redundant with ARP.")


if __name__ == "__main__":
    main()
