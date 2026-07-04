"""Driver-name normalization and fuzzy matching (src/utils.py).

The matcher's contract (per its docstring): merge typos, accents, initials,
nicknames and known aliases of the SAME driver, while refusing to merge two
DIFFERENT drivers who share a first name or surname.
"""
import pytest

from src.utils import fuzzy_match_name, normalize_driver_name


# ── normalize_driver_name ────────────────────────────────────────────────

def test_normalize_folds_accents_to_ascii():
    assert normalize_driver_name("Daniel Suárez") == "daniel suarez"


def test_normalize_lowercases():
    assert normalize_driver_name("DANIEL SUÁREZ") == "daniel suarez"


def test_normalize_strips_jr_suffix_with_and_without_period():
    assert normalize_driver_name("Martin Truex Jr.") == "martin truex"
    assert normalize_driver_name("Ricky Stenhouse Jr") == "ricky stenhouse"


@pytest.mark.parametrize("suffix", ["Sr.", "II", "III", "IV"])
def test_normalize_strips_other_suffixes(suffix):
    assert normalize_driver_name(f"Sam Hornish {suffix}") == "sam hornish"


def test_normalize_strips_periods_in_initials():
    assert normalize_driver_name("A.J. Allmendinger") == "aj allmendinger"


def test_normalize_collapses_whitespace():
    assert normalize_driver_name("  Kyle   Busch ") == "kyle busch"


def test_normalize_empty_and_none():
    assert normalize_driver_name("") == ""
    assert normalize_driver_name(None) == ""


def test_normalize_applies_nemechek_aliases():
    # DRIVER_ALIASES in src/config.py maps the feed's middle-initial and
    # short forms onto the canonical "john hunter nemechek".
    canonical = normalize_driver_name("John Hunter Nemechek")
    assert canonical == "john hunter nemechek"
    assert normalize_driver_name("John H. Nemechek") == canonical
    assert normalize_driver_name("John Nemechek") == canonical
    assert normalize_driver_name("J.H. Nemechek") == canonical


def test_normalize_applies_stage_name_alias():
    assert normalize_driver_name("Cleetus McFarland") == "garrett mitchell"


# ── fuzzy_match_name: MUST match ─────────────────────────────────────────

CANDIDATES = [
    "Carson Kvapil", "Austin Hill", "Kyle Larson", "Daniel Suárez",
    "Nicholas Sanchez", "Ryan Blaney", "Cole Custer", "Zane Smith",
]


def test_matches_small_surname_misspelling():
    assert fuzzy_match_name("Carson Kvapili", CANDIDATES) == "Carson Kvapil"
    assert fuzzy_match_name("Ryan Blany", CANDIDATES) == "Ryan Blaney"


def test_matches_accent_variants_both_directions():
    assert fuzzy_match_name("Daniel Suarez", CANDIDATES) == "Daniel Suárez"
    assert fuzzy_match_name("Daniel Suárez", ["Daniel Suarez"]) == "Daniel Suarez"


def test_matches_nickname_first_name():
    assert fuzzy_match_name("Nick Sanchez", CANDIDATES) == "Nicholas Sanchez"


def test_matches_first_initial():
    assert fuzzy_match_name("A Hill", CANDIDATES) == "Austin Hill"


def test_matches_middle_initial_abbreviation_of_present_middle():
    # John H <-> John Hunter: both names HAVE a middle, one abbreviates it.
    assert fuzzy_match_name("John H Doe", ["John Hunter Doe"]) == "John Hunter Doe"


def test_matches_suffix_variants():
    assert fuzzy_match_name("Ricky Stenhouse Jr.", ["Ricky Stenhouse"]) == "Ricky Stenhouse"


# ── fuzzy_match_name: MUST NOT match ─────────────────────────────────────

def test_rejects_same_surname_different_first_name():
    assert fuzzy_match_name("Austin Hill", ["Timmy Hill"]) is None
    assert fuzzy_match_name("Chandler Smith", ["Zane Smith"]) is None
    # "Ed" expands to Edward, not Erik — still different drivers.
    assert fuzzy_match_name("Ed Jones", ["Erik Jones"]) is None


def test_rejects_same_first_name_different_surname():
    assert fuzzy_match_name("Cole Custer", ["Cole Butcher"]) is None


def test_rejects_middle_initial_present_vs_absent():
    # Documented rule: a middle in one name but not the other means
    # DIFFERENT drivers (Jason M White and Jason White are separate people).
    assert fuzzy_match_name("Jason M White", ["Jason White"]) is None
    assert fuzzy_match_name("Austin Hill", ["Austin J Hill"]) is None


def test_rejects_single_token_name_as_too_ambiguous():
    assert fuzzy_match_name("Larson", ["Kyle Larson"]) is None


def test_empty_inputs_return_none():
    assert fuzzy_match_name("", CANDIDATES) is None
    assert fuzzy_match_name("Kyle Larson", []) is None
    assert fuzzy_match_name(None, CANDIDATES) is None


# ── threshold behavior ───────────────────────────────────────────────────

def test_threshold_arg_is_inert_backward_compat_only():
    # The docstring says `threshold` is retained for backward compatibility
    # but superseded by the component-wise logic. A permissive threshold must
    # not create a bad merge, and a strict one must not block a good one.
    assert fuzzy_match_name("Austin Hill", ["Timmy Hill"], threshold=0.0) is None
    assert fuzzy_match_name("Carson Kvapili", ["Carson Kvapil"],
                            threshold=0.99) == "Carson Kvapil"
