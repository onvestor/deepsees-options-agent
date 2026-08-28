"""The six ``metrics.*`` acceptance bands, as prefilter rejection gates.

They were declared in config and read by nothing until 28 Aug 2026: the
prefilter gated on open interest, bid, spread percentage, delta, DTE and expiry
type, computed the metrics, and used them only for ranking and the log. A band
that is never applied is worse than an absent one, because the config reads as
though a rule is in force.

Two of the six had no computed field to gate on and are derived here rather
than assumed equivalent to a similarly-named metric:

* ``max_spread_cost_pct_of_premium`` names the spread against *premium*, while
  ``spread_cost_pct_of_atr`` measures it against the ATR-implied option move.
* ``max_breakeven_move_pct`` names a percentage move of the *underlying*, while
  ``breakeven_distance_atr`` is denominated in ATRs.

Gating either on the wrong one would silently reject a different population,
which is the failure these tests exist to prevent.
"""
from __future__ import annotations

import math

import pytest

from src.options.metrics import compute_metrics, modeled_hold_hours
from src.options.prefilter import REASONS
from tests.test_prefilter import (  # reuse the shared fixtures
    IN_BAND,
    SESSION,
    _pinned,
    calendar,
    evaluate,
    limits,
    quote,
    spec,
)

BAND_REASONS = (
    "theta too high",
    "gamma too low",
    "iv rich",
    "spread cost vs premium",
    "breakeven too far",
    "modeled pnl too low",
)


def _bands(limits, **overrides):
    """Bands on, with every threshold set permissively unless overridden.

    Each test then tightens exactly one. A test that tightened a band while
    another happened to bind would assert the wrong gate.
    """
    base = {
        "prefilter.apply_metric_bands": True,
        "metrics.max_theta_pct_per_day": 10.0,
        "metrics.min_gamma_per_1pct": 0.0,
        "metrics.max_iv_vs_rv_ratio": 100.0,
        "metrics.max_spread_cost_pct_of_premium": 10.0,
        "metrics.max_breakeven_move_pct": 10.0,
        "metrics.min_modeled_pnl_ratio": -1e9,
    }
    base.update(overrides)
    return _pinned(limits, **base)


# --- the closed reason vocabulary ------------------------------------------


def test_every_band_reason_is_in_the_closed_set():
    """REASONS is documented as a closed set a dashboard can rely on."""
    for reason in BAND_REASONS:
        assert reason in REASONS


# --- the bands are off unless asked for ------------------------------------


def test_bands_off_leaves_the_survivor_untouched(calendar, limits):
    """The flag exists so the gates can be measured before being shipped."""
    tight = _bands(limits, **{"metrics.max_theta_pct_per_day": 0.0})
    off = _pinned(tight, **{"prefilter.apply_metric_bands": False})
    [candidate] = evaluate([spec()], [quote()], calendar, off)
    assert candidate.survived


def test_bands_on_can_reject_the_same_contract(calendar, limits):
    tight = _bands(limits, **{"metrics.max_theta_pct_per_day": 0.0})
    [candidate] = evaluate([spec()], [quote()], calendar, tight)
    assert not candidate.survived
    assert "theta too high" in candidate.failures


# --- each band, one at a time ----------------------------------------------


def test_theta_band_rejects_fast_decay(calendar, limits):
    # theta 0.20 on a 2.00 premium = 10% per day.
    tight = _bands(limits, **{"metrics.max_theta_pct_per_day": 0.05})
    [candidate] = evaluate([spec()], [quote(theta=-0.20)], calendar, tight)
    assert "theta too high" in candidate.failures


def test_gamma_band_rejects_an_unresponsive_contract(calendar, limits):
    tight = _bands(limits, **{"metrics.min_gamma_per_1pct": 1.0})
    [candidate] = evaluate([spec()], [quote(gamma=0.001)], calendar, tight)
    assert "gamma too low" in candidate.failures


def test_gamma_band_is_measured_per_1pct_not_raw(calendar, limits):
    """Raw gamma is not comparable across price levels, which is the whole
    reason gamma_per_1pct exists. The key is named for the scaled metric."""
    # gamma 0.04 at spot 100 -> gamma_per_1pct = 0.04 * 1.0 = 0.04.
    tight = _bands(limits, **{"metrics.min_gamma_per_1pct": 0.03})
    [candidate] = evaluate([spec()], [quote(gamma=0.04)], calendar, tight)
    assert "gamma too low" not in candidate.failures

    tighter = _bands(limits, **{"metrics.min_gamma_per_1pct": 0.05})
    [candidate] = evaluate([spec()], [quote(gamma=0.04)], calendar, tighter)
    assert "gamma too low" in candidate.failures


def test_iv_band_rejects_rich_premium(calendar, limits):
    # iv 0.30 against rv 0.25 = 1.2.
    tight = _bands(limits, **{"metrics.max_iv_vs_rv_ratio": 1.1})
    [candidate] = evaluate([spec()], [quote(iv=0.30)], calendar, tight)
    assert "iv rich" in candidate.failures


def test_spread_band_is_redundant_with_the_structural_spread_gate(calendar, limits):
    """``max_spread_cost_pct_of_premium`` cannot fire, and that is the finding.

    Premium is the mid, so ``spread_pct_of_premium`` is arithmetically the same
    number as the prefilter's ``spread_pct_of_mid``. The structural gate runs
    first, so whichever threshold is tighter does all the work. Live on 28 Aug
    this band rejected zero contracts across nine symbols -- not because
    spreads were tight, but because it is unreachable.

    Asserted rather than deleted so that a future change which makes the metric
    band tighter than the structural one fails here loudly.
    """
    # spread 0.20 on a 2.00 mid = 10%, over the structural gate either way.
    q = quote(bid=1.90, ask=2.10)
    tight = _bands(limits, **{"metrics.max_spread_cost_pct_of_premium": 0.05})
    [candidate] = evaluate([spec()], [q], calendar, tight)

    assert "spread" in candidate.failures            # the structural gate
    assert "spread cost vs premium" not in candidate.failures
    assert candidate.metrics is None                 # never scored, so never banded


def test_the_two_spread_measures_are_the_same_number():
    """The identity behind the redundancy, asserted directly."""
    m = _metrics(bid=1.90, ask=2.10)
    mid = (1.90 + 2.10) / 2
    assert m.spread_pct_of_premium == pytest.approx((2.10 - 1.90) / mid)


def test_breakeven_band_is_a_pct_of_spot(calendar, limits):
    """strike 100 + ask 2.03 = 102.03 breakeven, 2.03% above a spot of 100."""
    tight = _bands(limits, **{"metrics.max_breakeven_move_pct": 0.01})
    loose = _bands(limits, **{"metrics.max_breakeven_move_pct": 0.03})
    assert "breakeven too far" in evaluate([spec()], [quote()], calendar, tight)[0].failures
    assert "breakeven too far" not in evaluate([spec()], [quote()], calendar, loose)[0].failures


def test_a_contract_already_past_breakeven_clears_the_band(calendar, limits):
    """The distance is signed. Taking its absolute value would reject the best
    case in the set as though it were the worst."""
    tight = _bands(limits, **{"metrics.max_breakeven_move_pct": 0.001})
    # Deep ITM: strike 90, breakeven 92.03, spot 100 -> already past it.
    [candidate] = evaluate(
        [spec(strike=90.0)], [quote()], calendar, tight, spot=100.0
    )
    assert "breakeven too far" not in candidate.failures
    assert candidate.metrics.breakeven_move_pct < 0


def test_modeled_pnl_band_rejects_a_thin_edge(calendar, limits):
    tight = _bands(limits, **{"metrics.min_modeled_pnl_ratio": 1e9})
    [candidate] = evaluate([spec()], [quote()], calendar, tight)
    assert "modeled pnl too low" in candidate.failures


# --- the bands behave like every other reason ------------------------------


def test_a_band_reject_records_how_close_it_came(calendar, limits):
    """Bands are tunable thresholds, so the near-boundary report -- which is
    what a threshold change is decided from -- must cover them too."""
    tight = _bands(limits, **{"metrics.max_breakeven_move_pct": 0.02})
    [candidate] = evaluate([spec()], [quote()], calendar, tight)
    assert candidate.failures == ("breakeven too far",)
    assert candidate.boundary_distance is not None
    assert 0.0 < candidate.boundary_distance < 0.10


def test_metrics_are_kept_on_a_band_rejected_candidate(calendar, limits):
    """The values are the evidence for tuning the band that rejected it."""
    tight = _bands(limits, **{"metrics.max_theta_pct_per_day": 0.001})
    [candidate] = evaluate([spec()], [quote()], calendar, tight)
    assert not candidate.survived
    assert candidate.metrics is not None


def test_bands_run_last_and_do_not_mask_structural_rejects(calendar, limits):
    """A contract with no delta must still fail for that, not for a band --
    the bands need metrics, which an unscoreable contract does not have."""
    tight = _bands(limits, **{"metrics.max_theta_pct_per_day": 0.001})
    [candidate] = evaluate([spec()], [quote(delta=None)], calendar, tight)
    assert "no delta" in candidate.failures
    assert not any(r in candidate.failures for r in BAND_REASONS)


# --- the derived quantities ------------------------------------------------


def _metrics(**kw):
    base = dict(
        option_type="call", strike=100.0, spot=100.0, atr=2.0,
        bid=1.90, ask=2.10, delta=IN_BAND, gamma=0.04, theta=-0.20,
        implied_volatility=0.30, realized_vol=0.25,
        hold_hours=72.0, theta_day_hours=24.0,
    )
    base.update(kw)
    return compute_metrics(**base)


def test_spread_pct_of_premium_differs_from_spread_cost_pct_of_atr():
    """If these were the same number, one of the two bands would be a
    duplicate and the config would be describing a rule it does not have."""
    m = _metrics()
    assert m.spread_pct_of_premium == pytest.approx(0.20 / 2.00)
    assert m.spread_cost_pct_of_atr != pytest.approx(m.spread_pct_of_premium)


def test_breakeven_move_pct_is_the_distance_over_spot():
    m = _metrics()
    assert m.breakeven_move_pct == pytest.approx((100.0 + 2.10 - 100.0) / 100.0)


def test_derived_quantities_are_finite_on_a_normal_contract():
    m = _metrics()
    assert math.isfinite(m.spread_pct_of_premium)
    assert math.isfinite(m.breakeven_move_pct)


# --- the hold window -------------------------------------------------------


def test_modeled_hold_hours_is_derived_not_configured(limits):
    """There is deliberately no metrics.modeled_hold_hours key. It stood beside
    modeled_hold_sessions and disagreed with it -- 4.0 hours against a
    3-session hold -- which understated theta by a factor of eighteen."""
    from src.config import ConfigError

    with pytest.raises(ConfigError, match="modeled_hold_hours"):
        limits.get_float("metrics.modeled_hold_hours")

    sessions = limits.get_int("metrics.modeled_hold_sessions")
    day = limits.get_float("metrics.theta_day_hours")
    assert modeled_hold_hours(limits) == pytest.approx(sessions * day)


def test_the_example_config_does_not_reintroduce_the_key():
    """The example is what a fresh clone fills in. A key there that the code no
    longer reads would be filled in by an operator and silently ignored.

    Skips when ``config/`` is absent. That directory is gitignored apart from
    the committed ``*.example.yaml``, and the suite is required to run without
    it -- a test that hard-failed here would break the fresh-clone property it
    is meant to protect.
    """
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "config" / "limits.example.yaml"
    if not example.is_file():
        pytest.skip("config/limits.example.yaml absent from this checkout")

    text = example.read_text(encoding="utf-8")
    assert "modeled_hold_hours" not in text
    assert "modeled_hold_sessions" in text
