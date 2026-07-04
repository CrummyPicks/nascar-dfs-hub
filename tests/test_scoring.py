"""DFS scoring math — calc_dk_points / calc_fd_points vs the config tables.

DK Classic: finish points table + 1.0/place differential + 0.25/lap led
+ 0.45/fastest lap.
FanDuel: finish points table + 0.5/place differential + 0.1/lap led
+ 0.1/lap COMPLETED — and NO fastest-laps component at all.
"""
import inspect

import pytest

from src.config import (
    DK_FINISH_POINTS,
    FD_FINISH_POINTS,
    FD_PTS_LAPS_COMPLETED,
    FD_PTS_LAPS_LED,
    FD_PTS_PLACE_DIFF,
)
from src.utils import calc_dk_points, calc_fd_points


# ── Finish-points tables (config) ────────────────────────────────────────

def test_dk_finish_table_key_positions():
    # Documented DK Classic finish payouts at the tier boundaries.
    assert DK_FINISH_POINTS[1] == 45
    assert DK_FINISH_POINTS[2] == 42
    assert DK_FINISH_POINTS[11] == 32
    assert DK_FINISH_POINTS[21] == 21
    assert DK_FINISH_POINTS[31] == 10
    assert DK_FINISH_POINTS[40] == 1        # last paid position
    assert set(DK_FINISH_POINTS) == set(range(1, 41))


def test_fd_finish_table_matches_documented_curve():
    # Config docstring: 1st=43, 2nd=40, 3rd=38, then -1 per spot to 40th=1.
    assert FD_FINISH_POINTS[1] == 43
    assert FD_FINISH_POINTS[2] == 40
    assert FD_FINISH_POINTS[3] == 38
    assert FD_FINISH_POINTS[4] == 37
    for pos in range(4, 41):
        assert FD_FINISH_POINTS[pos] == 41 - pos
    assert FD_FINISH_POINTS[40] == 1
    assert set(FD_FINISH_POINTS) == set(range(1, 41))


# ── DK component scoring ─────────────────────────────────────────────────

@pytest.mark.parametrize("pos,expected", [
    (1, 45), (2, 42), (11, 32), (21, 21), (31, 10), (40, 1),
])
def test_dk_finish_points_only(pos, expected):
    # start == finish, no laps: score is purely the finish payout.
    assert calc_dk_points(pos, pos, 0, 0) == expected


def test_dk_place_differential_is_one_point_per_position():
    # Started 20th, finished 10th: +10 on top of the P10 payout.
    assert calc_dk_points(10, 20, 0, 0) == DK_FINISH_POINTS[10] + 10.0
    # Started 5th, finished 15th: -10.
    assert calc_dk_points(15, 5, 0, 0) == DK_FINISH_POINTS[15] - 10.0


def test_dk_mid_pack_neutral_differential():
    assert calc_dk_points(18, 18, 0, 0) == DK_FINISH_POINTS[18]


def test_dk_laps_led_quarter_point_per_lap():
    base = calc_dk_points(10, 10, 0, 0)
    assert calc_dk_points(10, 10, 40, 0) == base + 40 * 0.25


def test_dk_fastest_laps_045_per_lap():
    base = calc_dk_points(10, 10, 0, 0)
    assert calc_dk_points(10, 10, 0, 20) == base + 20 * 0.45


def test_dk_composite_win_from_pole_dominator():
    # P1 (45) + 0 diff + 100 led (25.0) + 40 fastest (18.0) = 88.0
    assert calc_dk_points(1, 1, 100, 40) == 88.0


def test_dk_composite_charger():
    # P2 from 10th: 42 + 8 diff + 30 led (7.5) + 12 fastest (5.4) = 62.9
    assert calc_dk_points(2, 10, 30, 12) == 62.9


def test_dk_accepts_numeric_strings():
    assert calc_dk_points("1", "1", "100", "40") == 88.0


def test_dk_invalid_input_returns_zero():
    assert calc_dk_points("DNF", 5, 0, 0) == 0.0
    assert calc_dk_points(None, 5, 0, 0) == 0.0


def test_dk_finish_outside_table_gets_no_finish_points():
    # 41st+ isn't in the table: only the differential remains.
    assert calc_dk_points(41, 41, 0, 0) == 0.0
    assert calc_dk_points(41, 43, 0, 0) == 2.0


# ── FD component scoring ─────────────────────────────────────────────────

def test_fd_signature_has_no_fastest_laps_parameter():
    # FanDuel does not pay fastest laps; the function must not even accept
    # them. The 4th arg is laps COMPLETED (default 0).
    params = inspect.signature(calc_fd_points).parameters
    assert list(params) == ["finish", "start", "laps_led", "laps_completed"]
    assert params["laps_completed"].default == 0
    assert not any("fastest" in p for p in params)


@pytest.mark.parametrize("pos,expected", [
    (1, 43), (2, 40), (3, 38), (4, 37), (20, 21), (40, 1),
])
def test_fd_finish_points_only(pos, expected):
    assert calc_fd_points(pos, pos, 0, 0) == expected


def test_fd_place_differential_is_half_point_per_position():
    assert calc_fd_points(10, 20, 0, 0) == FD_FINISH_POINTS[10] + 10 * 0.5
    assert calc_fd_points(20, 10, 0, 0) == FD_FINISH_POINTS[20] - 10 * 0.5


def test_fd_laps_led_tenth_point_per_lap():
    base = calc_fd_points(10, 10, 0, 0)
    assert calc_fd_points(10, 10, 50, 0) == pytest.approx(base + 5.0)


def test_fd_laps_completed_tenth_point_per_lap():
    base = calc_fd_points(10, 10, 0, 0)
    assert calc_fd_points(10, 10, 0, 267) == pytest.approx(base + 26.7)


def test_fd_laps_completed_defaults_to_zero():
    assert calc_fd_points(1, 1, 0) == 43.0


def test_fd_composite_win_with_laps():
    # P4 start -> win: 43 + 1.5 diff + 100 led (10.0) + 200 comp (20.0) = 74.5
    assert calc_fd_points(1, 4, 100, 200) == 74.5


def test_fd_rates_match_config_constants():
    # calc_fd_points hardcodes its per-lap/diff rates; keep them in lockstep
    # with the FD_PTS_* constants in src/config.py.
    base = calc_fd_points(10, 10, 0, 0)
    assert calc_fd_points(10, 10, 10, 0) - base == pytest.approx(10 * FD_PTS_LAPS_LED)
    assert calc_fd_points(10, 10, 0, 10) - base == pytest.approx(10 * FD_PTS_LAPS_COMPLETED)
    assert calc_fd_points(9, 10, 0, 0) - FD_FINISH_POINTS[9] == pytest.approx(FD_PTS_PLACE_DIFF)


def test_fd_invalid_input_returns_zero():
    assert calc_fd_points("DNF", 5, 0, 0) == 0.0
    assert calc_fd_points(None, 5, 0, 0) == 0.0
