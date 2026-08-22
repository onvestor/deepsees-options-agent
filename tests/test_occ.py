"""OCC build/parse, tested against recorded live Alpaca contracts.

Per CLAUDE.md: parsers are tested against recorded responses, not hand-written
JSON, because the shape surprises are the whole point. The critical assertion
is that our parsed root equals Alpaca's own ``root_symbol`` field -- we never
have to trust our own splitting rule.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.options.occ import TAIL_LENGTH, OccError, build, parse

FIXTURE = Path(__file__).parent / "fixtures" / "option_contracts.json"


def load_contracts() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["contracts"]


CONTRACTS = load_contracts()


def test_fixture_is_populated():
    assert len(CONTRACTS) > 100, "fixture should hold a broad sample"


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda c: c["symbol"])
def test_parse_matches_alpacas_own_fields(contract):
    """Every parsed field must agree with what Alpaca reports separately."""
    parsed = parse(contract["symbol"])

    assert parsed.root == contract["root_symbol"]
    assert parsed.expiry == date.fromisoformat(contract["expiration_date"])
    assert parsed.option_type == contract["type"]
    assert parsed.strike == pytest.approx(float(contract["strike_price"]))


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda c: c["symbol"])
def test_round_trip_is_byte_identical(contract):
    """build(parse(s)) == s, with no padding introduced or stripped."""
    symbol = contract["symbol"]
    parsed = parse(symbol)
    assert build(parsed.root, parsed.expiry, parsed.option_type, parsed.strike) == symbol


def test_roots_are_unpadded():
    """The whole reason this module exists: Alpaca does not OSI-pad the root."""
    for contract in CONTRACTS:
        assert " " not in contract["symbol"]
        assert len(contract["symbol"]) == len(contract["root_symbol"]) + TAIL_LENGTH


def test_osi_padded_form_is_rejected_as_different():
    symbol = "SPY260824C00500000"
    parsed = parse(symbol)
    osi = f"{parsed.root:<6}{parsed.expiry:%y%m%d}C00500000"
    assert osi != symbol
    with pytest.raises(OccError):
        parse(osi)  # embedded spaces make the root non-alphanumeric


# --- the cases the old non-greedy regex would have got wrong ---------------


@pytest.mark.parametrize(
    "symbol, root, strike",
    [
        ("AAPL1260918C00150000", "AAPL1", 150.0),   # adjusted contract, digit in root
        ("SPXW260918P04500000", "SPXW", 4500.0),    # weekly root
        ("A260918C00050000", "A", 50.0),            # single-character root
        ("BRKB260918C00400000", "BRKB", 400.0),
    ],
)
def test_fixed_width_split_handles_awkward_roots(symbol, root, strike):
    parsed = parse(symbol)
    assert parsed.root == root
    assert parsed.strike == pytest.approx(strike)
    assert build(parsed.root, parsed.expiry, parsed.option_type, parsed.strike) == symbol


def test_fractional_strike_rounds_rather_than_truncates():
    """17.50 must not become 00017499 through float error."""
    assert build("XYZ", date(2026, 9, 18), "call", 17.5).endswith("00017500")
    assert parse("XYZ260918C00017500").strike == pytest.approx(17.5)


def test_low_and_high_strikes():
    assert parse("XYZ260918C00000500").strike == pytest.approx(0.5)
    assert build("XYZ", date(2026, 9, 18), "put", 99999.999).endswith("99999999")


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "SPY",
        "SPY260824C0050000",          # 7 strike digits
        "SPY260824X00500000",         # bad option type
        "SPY261324C00500000",         # month 13
        "SPY260230C00500000",         # 30 February
        "SPY   260824C00500000",      # OSI padded
        "SPY-260824C00500000",        # punctuation in root
        "260824C00500000",            # no root
    ],
)
def test_malformed_symbols_raise(bad):
    with pytest.raises(OccError):
        parse(bad)


def test_parse_rejects_non_strings():
    with pytest.raises(OccError):
        parse(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "root, option_type, strike",
    [("", "call", 100.0), ("SPY", "straddle", 100.0), ("SPY", "call", 0.0), ("SPY", "call", -1.0)],
)
def test_build_rejects_bad_input(root, option_type, strike):
    with pytest.raises(OccError):
        build(root, date(2026, 9, 18), option_type, strike)  # type: ignore[arg-type]


def test_parse_is_case_and_whitespace_insensitive():
    assert parse("  spy260824c00500000  ") == parse("SPY260824C00500000")


def test_str_round_trips():
    symbol = "NVDA260824C00215000"
    assert str(parse(symbol)) == symbol
