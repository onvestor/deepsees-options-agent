"""Debit vertical metrics, and the x100 contract multiplier.

The multiplier tests are here because mixing per-share and per-contract
figures by a factor of 100 is the single easiest way to size a position 100x
wrong, and it is a mistake that reads perfectly plausibly in code review.
"""

from __future__ import annotations

import math

import pytest

from src.options.metrics import (
    CONTRACT_MULTIPLIER,
    MetricError,
    compute_metrics,
    compute_vertical_metrics,
)

# Long 100 call at 5.10, short 105 call at 2.90 -> 2.20 debit on a 5.00 wide
# spread. Round numbers throughout so every expectation is checkable by eye.
BASE = dict(
    option_type="call",
    long_strike=100.0,
    short_strike=105.0,
    spot=100.0,
    atr=2.0,
    long_bid=4.90, long_ask=5.10,
    short_bid=2.90, short_ask=3.10,
    long_delta=0.65, long_gamma=0.040, long_theta=-0.20,
    short_delta=0.30, short_gamma=0.030, short_theta=-0.15,
    hold_hours=4.0,
    theta_day_hours=24.0,
    breakeven_discount_k=1.0,
)


def vertical(**overrides):
    return compute_vertical_metrics(**{**BASE, **overrides})


# --- the four requested metrics --------------------------------------------


def test_max_risk_is_the_net_debit_in_dollars():
    """net debit = 5.10 ask - 2.90 bid = 2.20/share = $220 per contract."""
    result = vertical()
    assert result.net_debit == pytest.approx(2.20)
    assert result.max_risk_per_share == pytest.approx(2.20)
    assert result.max_risk == pytest.approx(220.00)


def test_max_gain_is_width_less_debit_in_dollars():
    """(5.00 width - 2.20 debit) = 2.80/share = $280 per contract."""
    result = vertical()
    assert result.max_gain_per_share == pytest.approx(2.80)
    assert result.max_gain == pytest.approx(280.00)


def test_max_risk_and_max_gain_sum_to_the_width():
    result = vertical()
    assert result.max_risk + result.max_gain == pytest.approx(result.width * CONTRACT_MULTIPLIER)


def test_reward_to_risk():
    """2.80 / 2.20 = 1.2727:1."""
    assert vertical().reward_to_risk == pytest.approx(2.80 / 2.20)


def test_pct_of_max_capturable_at_hold():
    """directional = 0.35*2 + 0.5*0.010*4 = 0.72
    decay       = -0.05 * (4/24) = -0.008333
    modeled     = 0.711666...  ->  / 2.80 max gain = 25.4%"""
    directional = 0.35 * 2.0 + 0.5 * 0.010 * 4.0
    decay = -0.05 * (4.0 / 24.0)
    assert vertical().pct_of_max_capturable_at_hold == pytest.approx((directional + decay) / 2.80)


def test_pct_capturable_is_the_metric_that_kills_a_short_hold_vertical():
    """A long-dated spread over a short hold barely moves: the short leg's
    decay offsets the long leg's gain and almost none of max gain is realised.

    This is the number that decides whether four bid-ask crossings are worth
    paying, and a high reward-to-risk does not rescue it.
    """
    sluggish = vertical(
        long_delta=0.60, short_delta=0.57,      # nearly delta-neutral net
        long_gamma=0.004, short_gamma=0.0039,
        long_theta=-0.02, short_theta=-0.019,
    )
    assert sluggish.reward_to_risk > 1.0
    assert sluggish.pct_of_max_capturable_at_hold < 0.05


def test_pct_capturable_is_capped_at_one():
    """A spread cannot be worth more than its width, however hard the model
    projects. An uncapped delta+gamma estimate on a narrow spread happily
    predicts a gain the structure cannot pay."""
    result = vertical(short_strike=101.0, short_bid=4.40, short_ask=4.60, atr=20.0)
    assert result.pct_of_max_capturable_at_hold <= 1.0
    assert result.modeled_pnl_1atr <= result.max_gain_per_share


def test_pct_capturable_can_be_negative():
    """Decay exceeding the directional gain is a real, reportable outcome."""
    result = vertical(
        long_delta=0.60, short_delta=0.599, long_gamma=0.001, short_gamma=0.001,
        long_theta=-0.90, short_theta=-0.10, hold_hours=12.0,
    )
    assert result.pct_of_max_capturable_at_hold < 0


# --- ranking ---------------------------------------------------------------


def test_rank_score_is_reward_to_risk_discounted_by_breakeven_atrs():
    """breakeven 102.20 is 1.10 ATRs away -> 1.2727 * exp(-1.0 * 1.10)."""
    result = vertical()
    assert result.breakeven == pytest.approx(102.20)
    assert result.breakeven_distance_atr == pytest.approx(1.10)
    assert result.rank_score == pytest.approx(result.reward_to_risk * math.exp(-1.10))


def test_the_discount_is_exponential_not_a_linear_divisor():
    """Regression on a real ranking bug.

    Reward-to-risk grows without bound as a spread moves OTM, so a 1/(1+kd)
    divisor cannot claw back a 21:1 ratio. Linear ranked the junk spread 5.6x
    ABOVE the good one; exponential puts it where it belongs.
    """
    solid = vertical()
    junk = vertical(
        long_strike=110.0, short_strike=120.0,
        long_bid=0.45, long_ask=0.55, short_bid=0.10, short_ask=0.20,
        long_delta=0.12, short_delta=0.04,
    )
    linear = lambda m: m.reward_to_risk / (1 + 1.0 * max(0.0, m.breakeven_distance_atr))
    assert linear(junk) > linear(solid)          # what the naive form did
    assert junk.rank_score < solid.rank_score    # what it does now


def test_a_nearer_breakeven_ranks_higher_at_equal_reward_to_risk():
    """The whole point of the discount: a 4:1 needing three ATRs is not better
    than a 2:1 needing half of one."""
    near = vertical(atr=8.0)     # same debit, breakeven is 0.275 ATRs away
    far = vertical(atr=1.0)      # ...and 2.2 ATRs away
    assert near.reward_to_risk == pytest.approx(far.reward_to_risk)
    assert near.rank_score > far.rank_score


def test_raw_reward_to_risk_would_have_preferred_the_wrong_spread():
    """A wide, cheap, far-OTM spread shows the best raw ratio and almost never
    pays. The discount is what stops it topping the ranking."""
    cheap_far = vertical(
        long_strike=110.0, short_strike=120.0,
        long_bid=0.45, long_ask=0.55, short_bid=0.10, short_ask=0.20,
        long_delta=0.12, short_delta=0.04,
    )
    solid_near = vertical()
    assert cheap_far.reward_to_risk > solid_near.reward_to_risk      # raw R:R prefers it
    assert cheap_far.rank_score < solid_near.rank_score              # discounted, it does not


def test_zero_discount_k_recovers_raw_reward_to_risk():
    result = vertical(breakeven_discount_k=0.0)
    assert result.rank_score == pytest.approx(result.reward_to_risk)


def test_breakeven_already_passed_is_not_penalised():
    """A deep-ITM spread whose breakeven sits below spot gets no discount, not
    a negative one."""
    result = vertical(long_strike=90.0, long_bid=10.4, long_ask=10.6,
                      short_bid=6.4, short_ask=6.6, short_strike=95.0)
    assert result.breakeven_distance_atr < 0
    assert result.rank_score == pytest.approx(result.reward_to_risk)


# --- the x100 contract multiplier ------------------------------------------


def test_contract_multiplier_is_one_hundred():
    assert CONTRACT_MULTIPLIER == 100


def test_per_contract_is_exactly_per_share_times_one_hundred():
    result = vertical()
    assert result.max_risk == pytest.approx(result.max_risk_per_share * 100)
    assert result.max_gain == pytest.approx(result.max_gain_per_share * 100)
    assert result.modeled_pnl_per_contract == pytest.approx(result.modeled_pnl_1atr * 100)


def test_a_two_dollar_twenty_debit_is_two_hundred_and_twenty_dollars():
    """Stated bluntly because this is the mistake: risking $2.20 and risking
    $220 differ by the whole account."""
    result = vertical()
    assert result.net_debit == pytest.approx(2.20)
    assert result.max_risk == pytest.approx(220.00)
    assert result.max_risk != pytest.approx(2.20)


def test_ratios_are_invariant_to_the_multiplier():
    """Reward-to-risk, capture and rank are per-share ratios -- the multiplier
    cancels. If any of them moves with it, a units bug has crept in."""
    one = vertical(multiplier=1)
    hundred = vertical(multiplier=100)
    thousand = vertical(multiplier=1000)
    for attribute in ("reward_to_risk", "pct_of_max_capturable_at_hold", "rank_score",
                      "breakeven_distance_atr", "net_delta", "modeled_pnl_1atr"):
        assert getattr(one, attribute) == pytest.approx(getattr(hundred, attribute))
        assert getattr(one, attribute) == pytest.approx(getattr(thousand, attribute))


def test_dollar_figures_do_scale_with_the_multiplier():
    ten = vertical(multiplier=10)
    hundred = vertical(multiplier=100)
    assert hundred.max_risk == pytest.approx(ten.max_risk * 10)
    assert hundred.max_gain == pytest.approx(ten.max_gain * 10)


def test_multiplier_is_echoed_for_audit():
    assert vertical().multiplier == 100
    assert vertical(multiplier=10).multiplier == 10


@pytest.mark.parametrize("contracts", [1, 3, 7])
def test_position_risk_scales_with_contract_count(contracts):
    """What the risk layer will actually multiply, stated explicitly."""
    result = vertical()
    assert result.max_risk * contracts == pytest.approx(220.00 * contracts)
    assert result.max_gain * contracts == pytest.approx(280.00 * contracts)


def test_single_leg_per_contract_figures_also_use_the_multiplier():
    result = compute_metrics(
        option_type="call", strike=100.0, spot=100.0, atr=2.0,
        bid=1.90, ask=2.10, delta=0.50, gamma=0.04, theta=-0.20,
        implied_volatility=0.30, realized_vol=0.25,
        hold_hours=4.0, theta_day_hours=24.0,
    )
    assert result.premium == pytest.approx(2.00)
    assert result.premium_per_contract == pytest.approx(200.00)
    assert result.cost_per_contract == pytest.approx(210.00)      # paying the ask
    assert result.max_risk == pytest.approx(210.00)               # long option: premium paid


def test_single_leg_ratios_are_unaffected_by_the_multiplier():
    """Sanity check on the other half of the same units boundary."""
    result = compute_metrics(
        option_type="call", strike=100.0, spot=100.0, atr=2.0,
        bid=1.90, ask=2.10, delta=0.50, gamma=0.04, theta=-0.20,
        implied_volatility=0.30, realized_vol=0.25,
        hold_hours=4.0, theta_day_hours=24.0,
    )
    assert result.theta_pct_per_day == pytest.approx(0.10)
    assert result.premium_per_contract / result.premium == pytest.approx(100.0)


# --- structure validation --------------------------------------------------


def test_put_vertical_buys_the_higher_strike():
    result = vertical(option_type="put", long_strike=105.0, short_strike=100.0,
                      long_delta=-0.65, short_delta=-0.30)
    assert result.net_debit == pytest.approx(2.20)
    assert result.breakeven == pytest.approx(102.80)              # 105 - 2.20
    assert result.net_delta == pytest.approx(0.35)


def test_put_vertical_breakeven_distance_is_measured_downward():
    result = vertical(option_type="put", long_strike=105.0, short_strike=100.0,
                      long_delta=-0.65, short_delta=-0.30, spot=104.0)
    assert result.breakeven_distance_atr == pytest.approx((104.0 - 102.80) / 2.0)


@pytest.mark.parametrize(
    "override, match",
    [
        ({"short_strike": 95.0}, "short_strike > long_strike"),
        ({"option_type": "put", "long_strike": 100.0, "short_strike": 105.0,
          "long_delta": -0.30, "short_delta": -0.65}, "short_strike < long_strike"),
        ({"long_bid": 1.9, "long_ask": 2.0, "short_bid": 3.0, "short_ask": 3.1}, "credit spread"),
        ({"long_ask": 9.0, "short_bid": 1.0}, "no achievable gain"),
        ({"long_delta": 0.30, "short_delta": 0.65}, "net delta"),
        ({"atr": 0.0}, "atr must be positive"),
        ({"spot": 0.0}, "spot must be positive"),
        ({"multiplier": 0}, "multiplier must be positive"),
        ({"long_bid": 0.0}, "unusable long quote"),
        ({"short_bid": 5.0, "short_ask": 1.0}, "unusable short quote"),
        ({"option_type": "butterfly"}, "unknown option_type"),
    ],
)
def test_invalid_structures_raise(override, match):
    with pytest.raises(MetricError, match=match):
        vertical(**override)


@pytest.mark.parametrize(
    "leg", ["long_delta", "long_gamma", "long_theta", "short_delta", "short_gamma", "short_theta"],
)
def test_missing_greek_on_either_leg_raises(leg):
    with pytest.raises(MetricError, match="not scoreable"):
        vertical(**{leg: None})


def test_credit_spreads_are_out_of_scope_by_construction():
    """Level 3 covers debit structures only; there is no code path to a credit."""
    with pytest.raises(MetricError, match="out of scope"):
        vertical(long_bid=2.80, long_ask=2.90, short_bid=3.10, short_ask=3.20)


# --- friction --------------------------------------------------------------


def test_round_trip_spread_cost_is_four_crossings():
    """0.10 half-spread on each leg = 0.20 to enter, 0.40 round trip."""
    result = vertical()
    assert result.entry_spread_cost == pytest.approx(0.20)
    assert result.round_trip_spread_cost == pytest.approx(0.40)


def test_round_trip_cost_against_max_gain_is_visible():
    """0.40 of friction against 2.80 max gain is 14% of the whole reward."""
    result = vertical()
    assert result.round_trip_spread_cost / result.max_gain_per_share == pytest.approx(0.1428, abs=1e-3)


def test_all_vertical_metrics_are_finite():
    result = vertical()
    assert result.is_finite
    assert not any(
        isinstance(v, float) and math.isnan(v) for v in result.as_dict().values()
    )
