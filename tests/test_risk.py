"""Risk engine: sizing, caps, kill switches.

The two acceptance criteria are property tests, not examples:

1. No combination of inputs produces a size exceeding any cap.
2. No cap is vacuously unbindable given live buying power.

The second exists because the first is satisfiable by a system whose caps are
all set so high they never bind -- which passes every property while providing
no protection at all.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from src.risk.caps import UNLIMITED, apply_caps, audit_caps, contracts_affordable
from src.risk.killswitch import (
    KillSwitchLimits,
    KillSwitchState,
    evaluate_kill_switches,
    fired_switches,
    is_halted,
)
from src.risk.sizing import AccountState, SizingLimits, compute_size


def _pinned(limits, **overrides):
    """Limits with specific keys pinned, so tuning config cannot break tests.

    The suite must assert against fixed numbers. Reading the operator's tuned
    ``config/limits.yaml`` made every legitimate tuning decision a test
    failure -- and those values are not even in a fresh clone.
    """
    from src.config import Section

    data = limits.as_dict()
    for dotted, value in overrides.items():
        node = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return Section(data, limits.source)


@pytest.fixture(scope="module")
def limits():
    from src.config import load_config

    # Pinned: the assertions below quote a 1% budget and a $1,500 premium cap.
    return SizingLimits.from_limits(_pinned(
        load_config().limits,
        **{"sizing.account_risk_pct_per_trade": 0.01,
           "sizing.max_premium_per_trade": 1500.0},
    ))


@pytest.fixture(scope="module")
def switch_limits():
    from src.config import load_config

    return KillSwitchLimits.from_limits(load_config().limits)


# --- strategies ------------------------------------------------------------

money = st.floats(min_value=1.0, max_value=5_000_000.0, allow_nan=False, allow_infinity=False)
contract_cost = st.floats(min_value=1.0, max_value=100_000.0, allow_nan=False, allow_infinity=False)
counts = st.integers(min_value=0, max_value=50)
multipliers = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)

accounts = st.builds(
    AccountState,
    equity=money,
    options_buying_power=money,
    open_premium=st.floats(min_value=0.0, max_value=1_000_000.0,
                           allow_nan=False, allow_infinity=False),
    open_positions=counts,
    positions_in_symbol=counts,
    entries_this_session=counts,
    entries_this_symbol_this_session=counts,
)

SLOW = settings(max_examples=400, deadline=None,
                suppress_health_check=[HealthCheck.function_scoped_fixture])


# --- acceptance 1: no input combination exceeds any cap --------------------


@given(cost=contract_cost, max_risk=contract_cost, account=accounts, multiplier=multipliers)
@SLOW
def test_no_combination_of_inputs_exceeds_any_cap(limits, cost, max_risk, account, multiplier):
    """The acceptance property, asserted against every cap simultaneously."""
    result = compute_size(cost, max_risk, account, limits, model_multiplier=multiplier)
    n = result.final_contracts

    assert n >= 0
    assert isinstance(n, int)
    if n == 0:
        return

    spend = n * cost
    tolerance = 1e-6

    assert n <= limits.max_contracts_per_trade
    assert spend <= limits.max_premium_per_trade + tolerance
    assert spend <= account.equity * limits.max_premium_pct_of_equity + tolerance
    assert spend <= account.options_buying_power + tolerance
    assert account.open_premium + spend <= limits.max_open_premium + tolerance
    assert (
        account.open_premium + spend
        <= account.equity * limits.max_open_premium_pct_of_equity + tolerance
    )
    assert account.open_positions < limits.max_concurrent_positions
    assert account.positions_in_symbol < limits.max_positions_per_symbol
    assert account.entries_this_session < limits.max_entries_per_session
    assert account.entries_this_symbol_this_session < limits.max_entries_per_symbol_per_session


@given(cost=contract_cost, max_risk=contract_cost, account=accounts, multiplier=multipliers)
@SLOW
def test_the_risk_layer_is_monotone(limits, cost, max_risk, account, multiplier):
    """Invariant 2: no code path by which a model increases exposure."""
    result = compute_size(cost, max_risk, account, limits, model_multiplier=multiplier)
    assert 0.0 <= result.model_multiplier <= 1.0
    assert result.after_model <= result.base_contracts
    assert result.final_contracts <= result.after_model or result.final_contracts == 0


@given(cost=contract_cost, max_risk=contract_cost, account=accounts,
       a=st.floats(0.0, 1.0), b=st.floats(0.0, 1.0))
@SLOW
def test_a_smaller_multiplier_never_yields_a_larger_size(limits, cost, max_risk, account, a, b):
    assume(a <= b)
    small = compute_size(cost, max_risk, account, limits, model_multiplier=a)
    large = compute_size(cost, max_risk, account, limits, model_multiplier=b)
    assert small.final_contracts <= large.final_contracts


@given(cost=contract_cost, max_risk=contract_cost, account=accounts)
@SLOW
def test_results_are_always_finite(limits, cost, max_risk, account):
    result = compute_size(cost, max_risk, account, limits)
    for value in (result.risk_budget, result.risk_per_contract, result.total_cost,
                  result.total_max_risk):
        assert math.isfinite(value)


@given(multiplier=multipliers)
def test_multiplier_is_always_clamped_into_range(limits, multiplier):
    """Shrink or veto only -- an out-of-range value is clamped and flagged."""
    account = AccountState(equity=100_000.0, options_buying_power=100_000.0)
    result = compute_size(1000.0, 1000.0, account, limits, model_multiplier=multiplier)
    assert 0.0 <= result.model_multiplier <= 1.0
    if not (0.0 <= multiplier <= 1.0):
        assert result.multiplier_clamped


# --- acceptance 2: no cap is vacuously unbindable --------------------------


def test_every_cap_can_bind_on_the_live_account():
    """Acceptance: caps must be reachable given real buying power.

    A cap above what the account can fund is inert -- the real constraint then
    lives at the broker and shows up as a rejection on the entry path.
    """
    from src.brokers.alpaca.client import build_clients, sizing_capital, with_retry
    from src.config import load_config

    config = load_config()
    clients = build_clients(config)
    account = with_retry(config, "get_account", clients.trading.get_account)

    limits = SizingLimits.from_limits(config.limits)
    audit = audit_caps(
        limits,
        equity=float(account.equity),
        options_buying_power=sizing_capital(account),
        cost_per_contract=184.0,
    )
    assert audit["all_caps_can_bind"], (
        "these caps can never bind and provide no protection: "
        f"{audit['unbindable']}"
    )


test_every_cap_can_bind_on_the_live_account = pytest.mark.live(
    test_every_cap_can_bind_on_the_live_account
)


def test_audit_detects_unbindable_caps(limits):
    """The offline half of the same check, on the account that motivated it.

    Equity $56,756 but only $530.72 of options buying power: every one of our
    caps sat above what could actually be spent.
    """
    audit = audit_caps(limits, equity=56_756.03, options_buying_power=530.72,
                       cost_per_contract=184.0)
    assert not audit["all_caps_can_bind"]
    assert "max_premium_per_trade" in audit["unbindable"]
    assert "max_open_premium" in audit["unbindable"]
    assert audit["spendable"] == pytest.approx(530.72)


def test_audit_passes_on_a_healthy_account(limits):
    audit = audit_caps(limits, equity=100_000.0, options_buying_power=100_000.0,
                       cost_per_contract=184.0)
    assert audit["all_caps_can_bind"]
    assert audit["unbindable"] == {}


@given(obp=st.floats(min_value=1.0, max_value=200.0))
@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_a_starved_account_makes_every_dollar_cap_unbindable(limits, obp):
    audit = audit_caps(limits, equity=100_000.0, options_buying_power=obp,
                       cost_per_contract=184.0)
    assert not audit["all_caps_can_bind"]


# --- sizing behaviour ------------------------------------------------------


def test_base_size_from_risk_budget(limits):
    """1% of $100k = $1,000 budget; $184 max risk per contract -> 5."""
    account = AccountState(equity=100_000.0, options_buying_power=100_000.0)
    result = compute_size(184.0, 184.0, account, limits)
    assert result.risk_budget == pytest.approx(1000.0)
    assert result.base_contracts == 5


def test_gap_risk_sizes_on_full_premium_not_the_stop(limits):
    """A swing hold cannot rely on the stop; the overnight gap can blow
    through it, so the risk carried is the whole premium."""
    account = AccountState(equity=100_000.0, options_buying_power=100_000.0)
    gapped = compute_size(184.0, 184.0, account, limits)
    assert gapped.gap_assumed is True
    assert gapped.risk_per_contract == pytest.approx(184.0)

    from dataclasses import replace

    stop_trusting = compute_size(
        184.0, 184.0, account, replace(limits, assume_stop_gapped=False)
    )
    assert stop_trusting.risk_per_contract < gapped.risk_per_contract
    assert stop_trusting.base_contracts > gapped.base_contracts


def test_size_always_rounds_down(limits):
    """$1,000 budget / $300 = 3.33 contracts -> 3, never 4."""
    account = AccountState(equity=100_000.0, options_buying_power=100_000.0)
    assert compute_size(300.0, 300.0, account, limits).base_contracts == 3


def test_below_min_contracts_is_no_trade(limits):
    """An expensive contract against a small budget is a veto, not a fraction."""
    account = AccountState(equity=10_000.0, options_buying_power=10_000.0)
    result = compute_size(50_000.0, 50_000.0, account, limits)
    assert result.final_contracts == 0
    assert not result.traded
    assert "below sizing.min_contracts" in (result.rejected_reason or "")


def test_options_buying_power_binds_before_the_broker_does(limits):
    """The cap that prevents an entry-path rejection."""
    account = AccountState(equity=100_000.0, options_buying_power=500.0)
    result = compute_size(184.0, 184.0, account, limits)
    assert result.final_contracts == 2                 # floor(500/184)
    assert "options_buying_power" in result.binding_caps


def test_open_premium_reduces_headroom(limits):
    account = AccountState(equity=100_000.0, options_buying_power=100_000.0,
                           open_premium=4_900.0)
    result = compute_size(184.0, 184.0, account, limits)
    assert result.final_contracts == 0 or "max_open_premium" in result.binding_caps


@pytest.mark.parametrize(
    "field, name",
    [
        ("open_positions", "max_concurrent_positions"),
        ("positions_in_symbol", "max_positions_per_symbol"),
        ("entries_this_session", "max_entries_per_session"),
        ("entries_this_symbol_this_session", "max_entries_per_symbol_per_session"),
    ],
)
def test_gate_caps_veto_entirely(limits, field, name):
    account = AccountState(equity=100_000.0, options_buying_power=100_000.0,
                           **{field: 99})
    result = compute_size(184.0, 184.0, account, limits)
    assert result.final_contracts == 0
    assert name in result.binding_caps


def test_every_cap_is_reported_whether_it_bound_or_not(limits):
    account = AccountState(equity=100_000.0, options_buying_power=100_000.0)
    result = compute_size(184.0, 184.0, account, limits)
    names = {v.name for v in result.caps}
    assert names == {
        "max_contracts_per_trade", "max_premium_per_trade", "max_premium_pct_of_equity",
        "options_buying_power", "max_open_premium", "max_open_premium_pct_of_equity",
        "max_concurrent_positions", "max_positions_per_symbol",
        "max_entries_per_session", "max_entries_per_symbol_per_session",
    }


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_degenerate_costs_are_rejected(limits, bad):
    account = AccountState(equity=100_000.0, options_buying_power=100_000.0)
    assert compute_size(bad, 184.0, account, limits).final_contracts == 0
    assert compute_size(184.0, bad, account, limits).final_contracts == 0


def test_contracts_affordable_rounds_down():
    assert contracts_affordable(1000.0, 300.0) == 3
    assert contracts_affordable(0.0, 300.0) == 0
    assert contracts_affordable(-5.0, 300.0) == 0
    assert contracts_affordable(1000.0, 0.0) == UNLIMITED


# --- kill switches ---------------------------------------------------------


def test_no_switch_fires_on_a_calm_session(switch_limits):
    state = KillSwitchState(start_of_day_equity=100_000.0, current_equity=100_500.0,
                            session_peak_equity=100_600.0)
    verdicts = evaluate_kill_switches(state, switch_limits)
    assert not is_halted(verdicts)
    assert fired_switches(verdicts) == ()


def test_daily_loss_pct_halts(switch_limits):
    state = KillSwitchState(start_of_day_equity=100_000.0, current_equity=96_000.0,
                            session_peak_equity=100_000.0)
    verdicts = evaluate_kill_switches(state, switch_limits)
    assert is_halted(verdicts)
    assert "daily_loss_halt_pct" in fired_switches(verdicts)


def test_drawdown_from_peak_can_fire_on_a_green_day(switch_limits):
    """A session up on the day can still have given back more than the limit.
    Measuring only from the open would miss it."""
    state = KillSwitchState(start_of_day_equity=100_000.0, current_equity=100_500.0,
                            session_peak_equity=110_000.0)
    verdicts = evaluate_kill_switches(state, switch_limits)
    assert is_halted(verdicts)
    assert "max_session_drawdown_pct" in fired_switches(verdicts)
    assert state.session_pnl > 0


def test_consecutive_losses_halt(switch_limits):
    state = KillSwitchState(start_of_day_equity=100_000.0, current_equity=99_900.0,
                            session_peak_equity=100_000.0, consecutive_losing_trades=99)
    assert "consecutive_losing_trades" in fired_switches(
        evaluate_kill_switches(state, switch_limits)
    )


def test_broker_error_streak_halts(switch_limits):
    state = KillSwitchState(start_of_day_equity=100_000.0, current_equity=100_000.0,
                            session_peak_equity=100_000.0, broker_error_streak=99)
    assert "broker_error_streak" in fired_switches(
        evaluate_kill_switches(state, switch_limits)
    )


def test_every_switch_is_evaluated_not_short_circuited(switch_limits):
    """Like the prefilter: how close the others came is what a review needs."""
    state = KillSwitchState(start_of_day_equity=100_000.0, current_equity=50_000.0,
                            session_peak_equity=100_000.0, consecutive_losing_trades=99,
                            broker_error_streak=99)
    verdicts = evaluate_kill_switches(state, switch_limits)
    assert len(verdicts) == 5
    assert all(v.fired for v in verdicts)


@given(
    start=st.floats(min_value=1000.0, max_value=1_000_000.0),
    current=st.floats(min_value=0.0, max_value=1_000_000.0),
    peak=st.floats(min_value=1000.0, max_value=2_000_000.0),
    losses=st.integers(0, 20),
    errors=st.integers(0, 20),
)
@settings(max_examples=300, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_kill_switches_are_total_and_deterministic(switch_limits, start, current, peak, losses, errors):
    state = KillSwitchState(start, current, peak, losses, errors)
    first = evaluate_kill_switches(state, switch_limits)
    second = evaluate_kill_switches(state, switch_limits)
    assert first == second
    assert len(first) == 5
    for verdict in first:
        assert isinstance(verdict.fired, bool)
        assert math.isfinite(verdict.observed)


@given(start=st.floats(min_value=1000.0, max_value=1_000_000.0),
       loss_fraction=st.floats(min_value=0.0, max_value=0.99))
@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_a_bigger_loss_never_un_fires_a_switch(switch_limits, start, loss_fraction):
    """Monotone in the direction that matters."""
    mild = KillSwitchState(start, start * (1 - loss_fraction / 2), start)
    severe = KillSwitchState(start, start * (1 - loss_fraction), start)
    mild_fired = set(fired_switches(evaluate_kill_switches(mild, switch_limits)))
    severe_fired = set(fired_switches(evaluate_kill_switches(severe, switch_limits)))
    assert mild_fired <= severe_fired


def test_killswitch_module_never_imports_the_agent_layer():
    """Structural: a kill switch exists for when the rest of the system is
    wrong, so it must not depend on the part that might be causing it."""
    source = Path(__file__).parent.parent / "src" / "risk" / "killswitch.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    for name in imported:
        assert not name.startswith("src.agents"), f"killswitch imports {name}"
        assert "anthropic" not in name, f"killswitch imports {name}"
