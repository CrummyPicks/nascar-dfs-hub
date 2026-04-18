"""End-to-end projection engine snapshot test.

Runs compute_projections() against a fixed synthetic race scenario with known
inputs and asserts the output matches a saved fixture. Catches unintended
changes to the projection math — e.g. the `sigs`-vs-`norm` bug that was
silently penalizing the wrong drivers.

Usage:
    python tests/test_projection_snapshot.py           # validates
    python tests/test_projection_snapshot.py --update  # regenerates fixture
                                                       # (use after intentional
                                                       # engine changes)

How it works:
    1. Build a deterministic input set: 15 synthetic drivers with varied
       track history, track-type history, qualifying positions, odds,
       and team signals. Mix of "has full data", "missing signals", and
       "no-odds longshots" to exercise the code paths.
    2. Call compute_projections() with fixed weights.
    3. Compare per-driver (proj_finish, proj_dk, laps_led, fast_laps)
       against the fixture JSON with 0.1 tolerance on floats.

Tolerance for floats: 0.1 (compute_projections output is rounded to 1 decimal
at most, so this allows for minor arithmetic ordering differences without
being blind to real math changes).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

# Allow running as `python tests/test_projection_snapshot.py` from repo root
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from src.projections import compute_projections

FIXTURE_PATH = Path(__file__).parent / "projection_snapshot.json"
FLOAT_TOLERANCE = 0.1


def build_inputs():
    """Build a deterministic synthetic race scenario.

    15 drivers modeled across archetypes:
    - Elite: full history, strong odds, top qualifier
    - Solid mid-pack: moderate history, decent odds
    - Value: cheap, good odds for salary, middling history
    - Rookie (no history): qualifying only, no odds quote (Baldwin archetype)
    - Backmarker: poor everything
    """
    drivers = [
        # Elites (top 3)
        "A Elite1", "A Elite2", "A Elite3",
        # Solid mid-pack
        "B Mid1", "B Mid2", "B Mid3", "B Mid4", "B Mid5",
        # Value plays
        "C Value1", "C Value2",
        # Backmarkers
        "D Back1", "D Back2",
        # No-history rookies (Baldwin archetype)
        "E Rookie1", "E Rookie2",
        # Lone contender without odds quote
        "F LongShot",
    ]
    field_size = len(drivers)

    # Track history (at this specific track): only veterans have it
    th_data = {
        "A Elite1": {"avg_running_pos": 6.5, "avg_finish": 8.0, "laps_led": 45, "races": 6},
        "A Elite2": {"avg_running_pos": 8.0, "avg_finish": 10.2, "laps_led": 30, "races": 5},
        "A Elite3": {"avg_running_pos": 10.5, "avg_finish": 12.5, "laps_led": 12, "races": 5},
        "B Mid1":   {"avg_running_pos": 14.0, "avg_finish": 15.0, "laps_led": 3, "races": 4},
        "B Mid2":   {"avg_running_pos": 16.0, "avg_finish": 17.5, "laps_led": 0, "races": 5},
        "B Mid3":   {"avg_running_pos": 18.0, "avg_finish": 19.5, "laps_led": 0, "races": 3},
        "C Value1": {"avg_running_pos": 20.5, "avg_finish": 18.0, "laps_led": 0, "races": 4},
        "D Back1":  {"avg_running_pos": 28.0, "avg_finish": 29.5, "laps_led": 0, "races": 5},
    }

    # Track type history (broader sample)
    tt_data = {
        "A Elite1": {"avg_running_pos": 7.0, "avg_finish": 8.5, "laps_led": 180, "fastest_laps": 32, "races": 18, "laps_led_per_race": 10.0},
        "A Elite2": {"avg_running_pos": 9.0, "avg_finish": 10.5, "laps_led": 120, "fastest_laps": 24, "races": 20, "laps_led_per_race": 6.0},
        "A Elite3": {"avg_running_pos": 11.5, "avg_finish": 13.0, "laps_led": 60, "fastest_laps": 15, "races": 16, "laps_led_per_race": 3.75},
        "B Mid1":   {"avg_running_pos": 13.5, "avg_finish": 15.0, "laps_led": 20, "fastest_laps": 8, "races": 15, "laps_led_per_race": 1.33},
        "B Mid2":   {"avg_running_pos": 15.5, "avg_finish": 17.0, "laps_led": 10, "fastest_laps": 5, "races": 18, "laps_led_per_race": 0.56},
        "B Mid3":   {"avg_running_pos": 17.0, "avg_finish": 19.0, "laps_led": 5, "fastest_laps": 3, "races": 14, "laps_led_per_race": 0.36},
        "B Mid4":   {"avg_running_pos": 18.0, "avg_finish": 20.0, "laps_led": 2, "fastest_laps": 3, "races": 10, "laps_led_per_race": 0.20},
        "B Mid5":   {"avg_running_pos": 19.0, "avg_finish": 21.0, "laps_led": 0, "fastest_laps": 2, "races": 12, "laps_led_per_race": 0.0},
        "C Value1": {"avg_running_pos": 19.5, "avg_finish": 18.5, "laps_led": 0, "fastest_laps": 2, "races": 14, "laps_led_per_race": 0.0},
        "C Value2": {"avg_running_pos": 22.0, "avg_finish": 22.0, "laps_led": 0, "fastest_laps": 1, "races": 9, "laps_led_per_race": 0.0},
        "D Back1":  {"avg_running_pos": 27.0, "avg_finish": 29.0, "laps_led": 0, "fastest_laps": 0, "races": 12, "laps_led_per_race": 0.0},
        "D Back2":  {"avg_running_pos": 29.0, "avg_finish": 30.5, "laps_led": 0, "fastest_laps": 0, "races": 8, "laps_led_per_race": 0.0},
        "F LongShot": {"avg_running_pos": 24.0, "avg_finish": 25.5, "laps_led": 0, "fastest_laps": 1, "races": 7, "laps_led_per_race": 0.0},
    }
    # Rookies and LongShot: no track-type data for rookies, some for LongShot

    # Qualifying positions — varied
    qual_pos = {
        "A Elite1": 2, "A Elite2": 4, "A Elite3": 8,
        "B Mid1": 6, "B Mid2": 10, "B Mid3": 14, "B Mid4": 16, "B Mid5": 18,
        "C Value1": 25, "C Value2": 12,
        "D Back1": 28, "D Back2": 30,
        "E Rookie1": 13, "E Rookie2": 22,
        "F LongShot": 24,
    }

    # Odds (implied finish positions — lower is better)
    odds_finish = {
        "A Elite1": 3.5, "A Elite2": 5.0, "A Elite3": 8.0,
        "B Mid1": 11.0, "B Mid2": 14.0, "B Mid3": 17.0, "B Mid4": 19.0, "B Mid5": 21.0,
        "C Value1": 23.0, "C Value2": 18.0,
        "D Back1": 29.0, "D Back2": 31.0,
        # Rookies and LongShot: no odds quoted (Vegas skipped them)
    }

    # No practice this race (common scenario — rained out)
    practice_data = {}

    # Team signal (from historical team stats — arbitrary values here)
    team_signal = {
        d: v for d, v in zip(drivers, [12.0, 13.0, 15.0, 17.0, 18.0, 19.0, 20.0, 21.0,
                                         22.0, 19.0, 27.0, 28.0, 24.0, 26.0, 25.0])
    }

    # Fixed weights — intermediate-track defaults
    wn = {
        "odds": 0.30, "track": 0.25, "track_type": 0.15,
        "practice": 0.0, "team": 0.15, "qual": 0.15,
    }

    return {
        "drivers": drivers,
        "field_size": field_size,
        "wn": wn,
        "th_data": th_data,
        "tt_data": tt_data,
        "qual_pos": qual_pos,
        "practice_data": practice_data,
        "odds_finish": odds_finish,
        "team_signal": team_signal,
        "race_laps": 400,
        "track_name": "Synthetic Intermediate",
        "track_type": "intermediate",
        "series_id": 1,
    }


def run_projection(inputs):
    """Run the projection engine and return a canonical dict for comparison."""
    proj_rows, proj_detail, signal_details = compute_projections(
        return_signal_details=True,
        drivers=inputs["drivers"],
        field_size=inputs["field_size"],
        wn=inputs["wn"],
        th_data=inputs["th_data"],
        tt_data=inputs["tt_data"],
        qual_pos=inputs["qual_pos"],
        practice_data=inputs["practice_data"],
        odds_finish=inputs["odds_finish"],
        odds_display={},
        team_signal=inputs["team_signal"],
        mfr_adjustment={},
        team_adj_data={},
        dnf_data={},
        race_laps=inputs["race_laps"],
        track_name=inputs["track_name"],
        track_type=inputs["track_type"],
        series_id=inputs["series_id"],
        calibration={},
        cross_th_lookup={},
    )

    result = {}
    for row in proj_rows:
        d = row["driver"]
        result[d] = {
            "proj_finish": round(row["proj_finish"], 2),
            "proj_dk": round(row["proj_dk"], 1),
            "laps_led": int(row["laps_led"]),
            "fast_laps": int(row["fast_laps"]),
            "diff_pts": round(row["diff_pts"], 1),
        }
    return result


def compare(actual, expected):
    """Return list of diff messages; empty list means match."""
    diffs = []
    all_drivers = set(actual) | set(expected)
    for d in sorted(all_drivers):
        a = actual.get(d)
        e = expected.get(d)
        if a is None:
            diffs.append(f"MISSING in actual: {d}")
            continue
        if e is None:
            diffs.append(f"EXTRA in actual: {d}")
            continue
        for k in ["proj_finish", "proj_dk", "diff_pts"]:
            if abs(a[k] - e[k]) > FLOAT_TOLERANCE:
                diffs.append(f"{d}.{k}: expected {e[k]}, got {a[k]} (delta {a[k] - e[k]:+.2f})")
        for k in ["laps_led", "fast_laps"]:
            if a[k] != e[k]:
                diffs.append(f"{d}.{k}: expected {e[k]}, got {a[k]}")
    return diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true",
                        help="Regenerate the fixture (use after intentional engine changes)")
    args = parser.parse_args()

    inputs = build_inputs()
    actual = run_projection(inputs)

    if args.update or not FIXTURE_PATH.exists():
        FIXTURE_PATH.write_text(json.dumps(actual, indent=2, sort_keys=True))
        print(f"Fixture {'updated' if FIXTURE_PATH.exists() else 'created'}: {FIXTURE_PATH}")
        print(f"Captured projections for {len(actual)} drivers.")
        return 0

    expected = json.loads(FIXTURE_PATH.read_text())
    diffs = compare(actual, expected)

    if not diffs:
        print(f"OK — projection snapshot matches ({len(actual)} drivers).")
        return 0

    print(f"FAIL — projection snapshot mismatch ({len(diffs)} differences):")
    for d in diffs[:20]:
        print(f"  {d}")
    if len(diffs) > 20:
        print(f"  ... and {len(diffs) - 20} more")
    print()
    print("If the engine change was intentional, regenerate with:")
    print("  python tests/test_projection_snapshot.py --update")
    return 1


if __name__ == "__main__":
    sys.exit(main())
