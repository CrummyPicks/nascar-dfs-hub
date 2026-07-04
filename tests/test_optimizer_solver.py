"""Optimal-lineup solver (tabs/tab_optimizer.py _solve_optimal).

Branch-and-bound knapsack: maximize the objective column subject to salary
cap and exact roster size. Every expected optimum here is recomputed by
brute force inside the test, so the fixtures can't drift out of sync.
"""
import itertools

import pytest

from tabs.tab_optimizer import _solve_optimal


def _driver(name, salary, proj, **extra):
    d = {"Driver": name, "DK Salary": salary, "Proj Score": proj}
    d.update(extra)
    return d


def _brute_force(drivers, cap, size, col="Proj Score"):
    """Exhaustive optimum: (best_score, frozenset_of_names) or (None, None)."""
    best_score, best_names = None, None
    for combo in itertools.combinations(drivers, size):
        if sum(d["DK Salary"] for d in combo) <= cap:
            score = sum(d[col] for d in combo)
            if best_score is None or score > best_score:
                best_score = score
                best_names = frozenset(d["Driver"] for d in combo)
    return best_score, best_names


EIGHT_DRIVERS = [
    _driver("A", 9800, 30.1),
    _driver("B", 8700, 24.3),
    _driver("C", 7600, 21.7),
    _driver("D", 6400, 19.2),
    _driver("E", 5300, 15.8),
    _driver("F", 4900, 12.4),
    _driver("G", 4200, 9.9),
    _driver("H", 3600, 5.3),
]


def test_matches_brute_force_optimum_8_drivers():
    cap, size = 15000, 3
    expected_score, expected_names = _brute_force(EIGHT_DRIVERS, cap, size)
    assert expected_score is not None  # fixture sanity: feasible lineup exists

    lineup = _solve_optimal(EIGHT_DRIVERS, cap, size)

    assert len(lineup) == size
    assert sum(d["DK Salary"] for d in lineup) <= cap
    assert sum(d["Proj Score"] for d in lineup) == pytest.approx(expected_score)
    assert {d["Driver"] for d in lineup} == set(expected_names)


def test_respects_salary_cap_over_raw_projection():
    # The three biggest projections (A+B+C = 26100) blow the cap; the solver
    # must find the best CAP-LEGAL trio instead of the best raw trio.
    cap, size = 15000, 3
    top3_salary = sum(d["DK Salary"] for d in EIGHT_DRIVERS[:3])
    assert top3_salary > cap  # fixture sanity

    lineup = _solve_optimal(EIGHT_DRIVERS, cap, size)
    assert sum(d["DK Salary"] for d in lineup) <= cap


def test_beats_greedy_by_value():
    # Greedy by points-per-dollar picks A then B (39.0 total) and strands
    # $2000; the true optimum pairs A with C for 46.0. Proves the solver
    # actually searches instead of ranking by value.
    drivers = [
        _driver("A", 4000, 20.0),  # 5.00 pts/$1k  <- greedy pick 1
        _driver("B", 4000, 19.0),  # 4.75          <- greedy pick 2
        _driver("C", 6000, 26.0),  # 4.33          <- the one greedy misses
        _driver("D", 6000, 25.0),
        _driver("E", 2000, 5.0),
    ]
    cap, size = 10000, 2

    # Greedy-by-value baseline, computed here so the trap is explicit.
    greedy, spent = [], 0
    for d in sorted(drivers, key=lambda d: d["Proj Score"] / d["DK Salary"],
                    reverse=True):
        if len(greedy) < size and spent + d["DK Salary"] <= cap:
            greedy.append(d)
            spent += d["DK Salary"]
    greedy_score = sum(d["Proj Score"] for d in greedy)
    assert greedy_score == 39.0  # fixture sanity: the trap is armed

    expected_score, expected_names = _brute_force(drivers, cap, size)
    lineup = _solve_optimal(drivers, cap, size)
    lineup_score = sum(d["Proj Score"] for d in lineup)

    assert lineup_score == pytest.approx(expected_score) == 46.0
    assert {d["Driver"] for d in lineup} == set(expected_names) == {"A", "C"}
    assert lineup_score > greedy_score


def test_objective_col_overrides_proj_score():
    # Opt Score (the cash/GPP objective) inverts the Proj Score ranking; the
    # solver must maximize whichever column it was told to.
    drivers = [
        _driver("X", 5000, 30.0, **{"Opt Score": 5.0}),
        _driver("Y", 5000, 10.0, **{"Opt Score": 20.0}),
        _driver("Z", 5000, 20.0, **{"Opt Score": 10.0}),
        _driver("W", 5000, 5.0, **{"Opt Score": 30.0}),
    ]
    cap, size = 10000, 2

    by_proj = _solve_optimal(drivers, cap, size)
    assert {d["Driver"] for d in by_proj} == {"X", "Z"}

    by_opt = _solve_optimal(drivers, cap, size, objective_col="Opt Score")
    assert {d["Driver"] for d in by_opt} == {"W", "Y"}


def test_infeasible_cap_returns_empty_lineup():
    # Current behavior: when no full roster fits under the cap, the solver
    # returns [] rather than a partial lineup.
    lineup = _solve_optimal(EIGHT_DRIVERS, 5000, 3)
    assert lineup == []
