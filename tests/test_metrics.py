"""The six metrics, with hand-computed expectations.

Every expected value below is derived from the definition by hand. Values are
per share; the 100x contract multiplier cancels in every ratio here.
"""

from __future__ import annotations

import math

import pytest

from src.options.metrics import (
    MetricError,
    compute_metrics,
    realized_volatility,
)

# A deliberately round contract so the arithmetic is checkable by inspection.
BASE = dict(
    option_type="call",
    strike=100.0,
    spot=100.0,
    atr=2.0,
    bid=1.90,
    ask=2.10,          # premium 2.00, spread 0.20
    delta=0.50,
    gamma=0.04,
    theta=-0.20,
    implied_volatility=0.30,
    realized_vol=0.25,
    hold_hours=4.0,
    theta_day_hours=24.0,
)


def metrics(**overrides):
    return compute_metrics(**{**BASE, **overrides})


# --- 1. theta as % of premium per day --------------------------------------


def test_theta_pct_per_day():
    """|theta| / premium = 0.20 / 2.00 = 10% of the premium per day."""
    assert metrics().theta_pct_per_day == pytest.approx(0.10)


def test_theta_sign_does_not_leak():
    """Theta arrives negative; the metric is a positive cost either way."""
    assert metrics(theta=-0.20).theta_pct_per_day == metrics(theta=0.20).theta_pct_per_day


# --- 2. gamma per 1% underlying move ---------------------------------------


def test_gamma_per_1pct():
    """gamma x 1% of spot = 0.04 x 1.00 = 0.04 delta gained per 1% move."""
    assert metrics().gamma_per_1pct == pytest.approx(0.04)


def test_gamma_per_1pct_scales_with_spot():
    """The point of the normalisation: comparable across a $70 and $700 name."""
    cheap = metrics(spot=50.0, strike=50.0)
    dear = metrics(spot=500.0, strike=500.0)
    assert dear.gamma_per_1pct == pytest.approx(10 * cheap.gamma_per_1pct)


# --- 3. IV vs realized -----------------------------------------------------


def test_iv_vs_rv_ratio():
    """0.30 / 0.25 = 1.2 -- implied is 20% rich to realized."""
    assert metrics().iv_vs_rv == pytest.approx(1.2)


def test_iv_equal_to_rv_is_one():
    assert metrics(implied_volatility=0.25).iv_vs_rv == pytest.approx(1.0)


def test_realized_volatility_of_a_flat_series_is_zero():
    assert realized_volatility([100.0] * 30, window=20) == pytest.approx(0.0)


def test_realized_volatility_matches_hand_computation():
    """Alternating +1%/-1% log returns: sd of returns x sqrt(252)."""
    prices, price = [100.0], 100.0
    for i in range(21):
        price *= 1.01 if i % 2 == 0 else (1 / 1.01)
        prices.append(price)
    result = realized_volatility(prices, window=20)

    returns = [math.log(prices[i + 1] / prices[i]) for i in range(len(prices) - 21, len(prices) - 1)]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    assert result == pytest.approx(math.sqrt(variance) * math.sqrt(252))


def test_realized_volatility_needs_enough_history():
    with pytest.raises(MetricError, match="need 21 positive closes"):
        realized_volatility([100.0] * 5, window=20)


def test_realized_volatility_rejects_tiny_window():
    with pytest.raises(MetricError):
        realized_volatility([100.0] * 30, window=1)


# --- 4. spread cost as % of the ATR-implied option move --------------------


def test_spread_cost_pct_of_atr():
    """One ATR moves the option delta x ATR = 0.5 x 2.0 = 1.00.
    The 0.20 spread is 20% of that."""
    assert metrics().spread_cost_pct_of_atr == pytest.approx(0.20)


def test_wide_spread_on_a_sluggish_contract_is_punished():
    """The metric that kills otherwise attractive contracts: a low-delta
    contract barely moves, so the same spread costs proportionally far more."""
    responsive = metrics(delta=0.60)
    sluggish = metrics(delta=0.15)
    assert sluggish.spread_cost_pct_of_atr > 3 * responsive.spread_cost_pct_of_atr


# --- 5. breakeven distance in ATRs -----------------------------------------


def test_breakeven_distance_atr_for_a_call():
    """ATM call: breakeven = 100 + 2.10 = 102.10, which is 1.05 ATRs away."""
    assert metrics().breakeven_distance_atr == pytest.approx(1.05)


def test_breakeven_distance_atr_for_a_put():
    """ATM put: breakeven = 100 - 2.10 = 97.90, again 1.05 ATRs of travel."""
    result = metrics(option_type="put", delta=-0.50)
    assert result.breakeven_distance_atr == pytest.approx(1.05)


def test_breakeven_distance_is_extrinsic_value_measured_in_atrs():
    """A long option's breakeven is always above spot by exactly its extrinsic
    value -- breakeven = strike + premium, so it can only sit below spot if the
    option trades under intrinsic, which is arbitrage.

    Deep ITM at strike 90, ask 11.10: intrinsic 10.00, extrinsic 1.10,
    so 1.10 / 2.00 ATR = 0.55 ATRs of travel needed.
    """
    result = metrics(strike=90.0, bid=10.9, ask=11.1, delta=0.9)
    intrinsic = 100.0 - 90.0
    extrinsic = 11.1 - intrinsic
    assert result.breakeven_distance_atr == pytest.approx(extrinsic / 2.0)
    assert result.breakeven_distance_atr == pytest.approx(0.55)


def test_deep_itm_needs_less_travel_than_atm():
    """Which is the whole reason the metric is worth computing."""
    deep_itm = metrics(strike=90.0, bid=10.9, ask=11.1, delta=0.9)
    assert deep_itm.breakeven_distance_atr < metrics().breakeven_distance_atr


def test_put_delta_sign_does_not_break_the_maths():
    result = metrics(option_type="put", delta=-0.50)
    assert result.spread_cost_pct_of_atr == pytest.approx(0.20)
    assert result.modeled_pnl_1atr == pytest.approx(metrics().modeled_pnl_1atr)


# --- 6. modeled P&L on a 1-ATR move over the hold --------------------------


def test_modeled_pnl_1atr():
    """directional = |d|*ATR + 0.5*gamma*ATR^2 = 1.00 + 0.5*0.04*4 = 1.08
    decay = 0.20 * (4/24) = 0.0333...  ->  1.0466..."""
    assert metrics().modeled_pnl_1atr == pytest.approx(1.08 - 0.20 * (4 / 24))


def test_longer_hold_costs_more_decay():
    assert metrics(hold_hours=8.0).modeled_pnl_1atr < metrics(hold_hours=4.0).modeled_pnl_1atr


def test_gamma_contributes_convexity():
    assert metrics(gamma=0.10).modeled_pnl_1atr > metrics(gamma=0.0).modeled_pnl_1atr


def test_decay_can_exceed_the_move():
    """A high-theta, low-delta contract loses money on a full ATR move."""
    result = metrics(delta=0.05, gamma=0.001, theta=-1.50, hold_hours=6.0)
    assert result.modeled_pnl_1atr < 0


# --- ranking key -----------------------------------------------------------


def test_pnl_to_spread_ratio_is_the_ranking_key():
    assert metrics().pnl_to_spread_ratio == pytest.approx(metrics().modeled_pnl_1atr / 0.20)


def test_ranking_prefers_the_tighter_spread_at_equal_edge():
    """Ranking on modeled P&L alone would favour contracts whose spread eats
    the gain; the ratio is what stops that."""
    tight = metrics(bid=1.95, ask=2.05)
    wide = metrics(bid=1.75, ask=2.25)
    assert tight.modeled_pnl_1atr == pytest.approx(wide.modeled_pnl_1atr)
    assert tight.pnl_to_spread_ratio > wide.pnl_to_spread_ratio


# --- fail closed -----------------------------------------------------------


@pytest.mark.parametrize("field", ["delta", "gamma", "theta", "implied_volatility"])
def test_missing_greek_raises_never_defaults(field):
    """The second line of the hard-reject defence."""
    with pytest.raises(MetricError, match="not scoreable"):
        metrics(**{field: None})


@pytest.mark.parametrize(
    "override",
    [
        {"atr": 0.0}, {"atr": -1.0}, {"spot": 0.0}, {"realized_vol": 0.0},
        {"bid": 0.0}, {"bid": 2.5, "ask": 2.0}, {"delta": 0.0},
        {"theta_day_hours": 0.0}, {"hold_hours": -1.0},
        {"option_type": "straddle"},
    ],
)
def test_degenerate_inputs_raise(override):
    with pytest.raises(MetricError):
        metrics(**override)


def test_all_metrics_are_finite():
    """Acceptance: every metric populated, no NaNs."""
    result = metrics()
    assert result.is_finite
    assert not any(math.isnan(v) for v in result.as_dict().values())


def test_inputs_are_echoed_for_audit():
    """A logged metric that cannot be recomputed later cannot be audited."""
    values = metrics().as_dict()
    for key in ("spot", "atr", "strike", "premium", "spread", "delta", "gamma", "theta",
                "implied_volatility", "realized_volatility", "hold_hours"):
        assert key in values
