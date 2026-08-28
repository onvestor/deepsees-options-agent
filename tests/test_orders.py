"""The order builder. Every Alpaca constraint, and the two order shapes.

No network. The trading client is a fake that records what it was asked to do,
because the value here is in *what request was constructed*, and a live test
can only tell you it was accepted.

The constraints in :func:`build_single_leg` are the ones CLAUDE.md warns about,
and each is asserted separately: a whole-number ``qty``, no ``notional``,
``time_in_force`` of ``day``, no ``extended_hours``, and an explicit
``position_intent``. They are worth individual tests because each fails at
submission with a message that does not name the field.
"""
from __future__ import annotations

from datetime import date, datetime

import pytest
from alpaca.trading.enums import OrderSide, OrderStatus, PositionIntent, TimeInForce

from src.brokers.alpaca.orders import (
    EXIT_URGENCY,
    ExecutionLimits,
    Intent,
    OrderError,
    Urgency,
    build_single_leg,
    cancel,
    is_filled,
    mid_price,
    place_entry,
    place_exit,
    poll_to_terminal,
    round_penny,
    step_ladder,
    urgency_for,
)
from src.brokers.alpaca.quotes import OptionQuote

SYMBOL = "SPY261016C00500000"


def quote(bid=1.90, ask=2.10, symbol=SYMBOL) -> OptionQuote:
    return OptionQuote(
        symbol=symbol, bid=bid, ask=ask, bid_size=10.0, ask_size=10.0,
        quote_ts=datetime(2026, 8, 28, 15, 0), delta=0.62, gamma=0.01,
        theta=-0.05, vega=0.4, rho=0.1, implied_volatility=0.22,
        last_trade_price=2.0, last_trade_ts=datetime(2026, 8, 28, 15, 0),
    )


# --- fakes -----------------------------------------------------------------


class FakeOrder:
    _next = 1

    def __init__(self, request, status=OrderStatus.FILLED, filled=None):
        FakeOrder._next += 1
        self.id = f"order-{FakeOrder._next}"
        self.symbol = request.symbol
        self.qty = request.qty
        self.side = request.side
        self.limit_price = request.limit_price
        self.time_in_force = request.time_in_force
        self.position_intent = request.position_intent
        self.status = status
        self.filled_qty = request.qty if filled is None else filled
        self.filled_avg_price = request.limit_price


class FakeTrading:
    """Records requests. ``script`` decides each order's outcome."""

    def __init__(self, script=None, account_equity=100_000.0):
        self.requests = []
        self.orders = {}
        self.cancelled = []
        self.script = list(script or [])
        self.account_equity = account_equity

    def submit_order(self, request):
        self.requests.append(request)
        if self.script:
            status, filled = self.script.pop(0)
        else:
            status, filled = OrderStatus.FILLED, None
        order = FakeOrder(request, status, filled)
        self.orders[order.id] = order
        return order

    def get_order_by_id(self, order_id):
        return self.orders[str(order_id)]

    def cancel_order_by_id(self, order_id):
        self.cancelled.append(str(order_id))
        self.orders[str(order_id)].status = OrderStatus.CANCELED

    def get_account(self):
        class A:
            equity = self.account_equity
            options_buying_power = self.account_equity
            cash = self.account_equity
            multiplier = "1"
        return A()


class FakeClients:
    def __init__(self, config, trading=None):
        self.config = config
        self.trading = trading or FakeTrading()
        self.writes = []

    def assert_writable(self, action):
        self.writes.append(action)


@pytest.fixture(scope="module")
def config():
    from src.config import load_config

    return load_config()


@pytest.fixture
def clients(config):
    return FakeClients(config)


@pytest.fixture
def limits(config):
    return ExecutionLimits.from_limits(config.limits)


def no_sleep(_seconds):
    return None


class Ticker:
    """A monotonic clock that advances only when asked."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1.0
        return self.t


# --- the Alpaca constraints ------------------------------------------------


def test_a_valid_request_carries_every_required_field():
    request = build_single_leg(
        symbol=SYMBOL, qty=2, intent=Intent.BUY_TO_OPEN, limit_price=2.00
    )
    assert request.symbol == SYMBOL
    assert request.qty == 2
    assert request.side is OrderSide.BUY
    assert request.time_in_force is TimeInForce.DAY
    assert request.position_intent is PositionIntent.BUY_TO_OPEN
    assert request.limit_price == 2.00


def test_notional_is_never_populated():
    """Populating it alongside qty is rejected, and we never size in dollars."""
    request = build_single_leg(
        symbol=SYMBOL, qty=1, intent=Intent.BUY_TO_OPEN, limit_price=2.00
    )
    assert getattr(request, "notional", None) is None


def test_extended_hours_is_absent():
    request = build_single_leg(
        symbol=SYMBOL, qty=1, intent=Intent.BUY_TO_OPEN, limit_price=2.00
    )
    assert not getattr(request, "extended_hours", False)


def test_time_in_force_is_day_not_gtc():
    """A gtc entry can fill on a session whose thesis was formed days earlier."""
    request = build_single_leg(
        symbol=SYMBOL, qty=1, intent=Intent.SELL_TO_CLOSE, limit_price=2.00
    )
    assert request.time_in_force is TimeInForce.DAY


def test_a_fractional_qty_is_refused():
    with pytest.raises(OrderError, match="whole number"):
        build_single_leg(symbol=SYMBOL, qty=1.5, intent=Intent.BUY_TO_OPEN,
                         limit_price=2.00)


def test_a_bool_qty_is_refused():
    """bool is an int in Python, and True would submit as qty=1."""
    with pytest.raises(OrderError, match="whole number"):
        build_single_leg(symbol=SYMBOL, qty=True, intent=Intent.BUY_TO_OPEN,
                         limit_price=2.00)


@pytest.mark.parametrize("qty", [0, -1])
def test_a_non_positive_qty_is_refused(qty):
    with pytest.raises(OrderError, match="positive"):
        build_single_leg(symbol=SYMBOL, qty=qty, intent=Intent.BUY_TO_OPEN,
                         limit_price=2.00)


@pytest.mark.parametrize("price", [0.0, -1.0])
def test_a_non_positive_limit_is_refused(price):
    with pytest.raises(OrderError, match="positive"):
        build_single_leg(symbol=SYMBOL, qty=1, intent=Intent.BUY_TO_OPEN,
                         limit_price=price)


def test_a_sub_penny_limit_is_refused():
    with pytest.raises(OrderError, match="sub-penny"):
        build_single_leg(symbol=SYMBOL, qty=1, intent=Intent.BUY_TO_OPEN,
                         limit_price=2.005)


def test_sell_to_close_carries_the_right_intent():
    request = build_single_leg(
        symbol=SYMBOL, qty=1, intent=Intent.SELL_TO_CLOSE, limit_price=2.00
    )
    assert request.side is OrderSide.SELL
    assert request.position_intent is PositionIntent.SELL_TO_CLOSE


# --- pricing ---------------------------------------------------------------


def test_mid_price_is_the_midpoint():
    assert mid_price(quote(1.90, 2.10)) == 2.00


def test_a_one_sided_quote_has_no_mid_to_rest_at():
    with pytest.raises(OrderError, match="no two-sided quote"):
        mid_price(OptionQuote.missing(SYMBOL))


def test_round_penny_beats_float_error():
    assert round_penny(2.675) == 2.68
    assert round_penny(0.1 + 0.2) == 0.30


# --- the exit ladder -------------------------------------------------------


def test_a_ladder_starts_at_the_mid_and_ends_at_the_bid():
    prices = step_ladder(quote(1.90, 2.10), 3)
    assert prices[0] == 2.00
    assert prices[-1] == 1.90


def test_every_ladder_ends_at_the_bid_whatever_its_length():
    """A ladder that stopped short would leave an exit resting at a price
    nobody is obliged to pay."""
    for steps in range(1, 8):
        assert step_ladder(quote(1.90, 2.10), steps)[-1] == 1.90


def test_a_one_step_ladder_is_the_bid_outright():
    assert step_ladder(quote(1.90, 2.10), 1) == [1.90]


def test_a_ladder_walks_downward_monotonically():
    prices = step_ladder(quote(1.00, 3.00), 5)
    assert prices == sorted(prices, reverse=True)


def test_a_ladder_needs_at_least_one_step():
    with pytest.raises(OrderError, match="at least one step"):
        step_ladder(quote(), 0)


def test_a_ladder_needs_a_two_sided_quote():
    with pytest.raises(OrderError, match="two-sided"):
        step_ladder(OptionQuote.missing(SYMBOL), 3)


# --- urgency is keyed to the reason ---------------------------------------


def test_a_stop_is_urgent_and_a_target_is_not():
    """A stop is running risk; a target is banking profit. A ladder that
    treated them alike would either give back gains or sit through a loss."""
    assert urgency_for("stop") is Urgency.HIGH
    assert urgency_for("target") is Urgency.LOW


def test_an_unknown_reason_exits_fast():
    """Fails closed in the direction of leaving. A reason nobody has reasoned
    about is not one to be patient through."""
    assert urgency_for("something_new") is Urgency.HIGH


def test_every_deterministic_exit_reason_is_mapped():
    """The replay broker's reasons are the ones the live exit layer emits."""
    for reason in ("stop", "target", "max_hold", "expiry_week",
                   "agent_exit_now", "scale_out_half"):
        assert reason in EXIT_URGENCY


def test_urgency_changes_speed_not_destination(config, limits):
    """Higher urgency means fewer, faster rungs -- not a different final price."""
    fast = step_ladder(quote(), limits.exit_steps[Urgency.HIGH])
    slow = step_ladder(quote(), limits.exit_steps[Urgency.LOW])
    assert len(fast) < len(slow)
    assert fast[-1] == slow[-1]
    assert limits.exit_step_seconds[Urgency.HIGH] < limits.exit_step_seconds[Urgency.LOW]


# --- entry -----------------------------------------------------------------


def test_entry_rests_at_the_mid(clients, limits):
    result = place_entry(clients, symbol=SYMBOL, qty=1, quote=quote(1.90, 2.10),
                         limits=limits, sleep=no_sleep, clock=Ticker())
    [request] = clients.trading.requests
    assert request.limit_price == 2.00
    assert request.side is OrderSide.BUY
    assert result.complete


def test_entry_places_exactly_one_order(clients, limits):
    """It does not chase. Repricing is the entry manager's job on its own
    cadence, and its step ceiling is derived per contract."""
    clients.trading.script = [(OrderStatus.CANCELED, 0)]
    place_entry(clients, symbol=SYMBOL, qty=1, quote=quote(), limits=limits,
                sleep=no_sleep, clock=Ticker())
    assert len(clients.trading.requests) == 1


def test_entry_is_guarded_by_mock_mode(clients, limits):
    place_entry(clients, symbol=SYMBOL, qty=1, quote=quote(), limits=limits,
                sleep=no_sleep, clock=Ticker())
    assert "buy_to_open" in clients.writes


def test_an_unaffordable_entry_is_refused_before_submission(config, limits):
    clients = FakeClients(config, FakeTrading(account_equity=50.0))
    with pytest.raises(OrderError, match="sizing capital"):
        place_entry(clients, symbol=SYMBOL, qty=1, quote=quote(), limits=limits,
                    sleep=no_sleep, clock=Ticker())
    assert clients.trading.requests == []


# --- exit ------------------------------------------------------------------


def test_an_exit_that_fills_on_the_first_rung_places_one_order(clients, limits):
    result = place_exit(clients, symbol=SYMBOL, qty=2, quote=quote(),
                        reason="target", limits=limits, sleep=no_sleep, clock=Ticker())
    assert len(clients.trading.requests) == 1
    assert result.complete


def test_an_unfilled_exit_walks_the_ladder(config, limits):
    """Each rung cancels the last and re-places lower."""
    trading = FakeTrading(script=[(OrderStatus.CANCELED, 0)] * 4)
    clients = FakeClients(config, trading)
    result = place_exit(clients, symbol=SYMBOL, qty=1, quote=quote(1.90, 2.10),
                        reason="target", limits=limits, sleep=no_sleep, clock=Ticker())
    prices = [r.limit_price for r in trading.requests]
    assert len(prices) == limits.exit_steps[Urgency.LOW]
    assert prices == sorted(prices, reverse=True)
    assert prices[-1] == 1.90
    assert not result.complete


def test_a_high_urgency_exit_reaches_the_bid_in_fewer_rungs(config, limits):
    trading = FakeTrading(script=[(OrderStatus.CANCELED, 0)] * 4)
    clients = FakeClients(config, trading)
    place_exit(clients, symbol=SYMBOL, qty=1, quote=quote(), reason="stop",
               limits=limits, sleep=no_sleep, clock=Ticker())
    assert len(trading.requests) == limits.exit_steps[Urgency.HIGH]


def test_a_partial_fill_re_places_only_the_remainder(config, limits):
    """Re-placing the full quantity would over-sell a partially filled
    position, which on a long option means going short."""
    trading = FakeTrading(script=[(OrderStatus.CANCELED, 1), (OrderStatus.FILLED, 2)])
    clients = FakeClients(config, trading)
    result = place_exit(clients, symbol=SYMBOL, qty=3, quote=quote(),
                        reason="stop", limits=limits, sleep=no_sleep, clock=Ticker())
    assert [r.qty for r in trading.requests] == [3, 2]
    assert result.complete


def test_the_ladder_stops_once_the_position_is_flat(config, limits):
    trading = FakeTrading(script=[(OrderStatus.FILLED, 2)])
    clients = FakeClients(config, trading)
    place_exit(clients, symbol=SYMBOL, qty=2, quote=quote(), reason="target",
               limits=limits, sleep=no_sleep, clock=Ticker())
    assert len(trading.requests) == 1


def test_a_re_quote_moves_only_the_remaining_rungs(config, limits):
    """A slow ladder against a stale quote walks to a bid that moved."""
    trading = FakeTrading(script=[(OrderStatus.CANCELED, 0)] * 4)
    clients = FakeClients(config, trading)
    place_exit(
        clients, symbol=SYMBOL, qty=1, quote=quote(1.90, 2.10), reason="target",
        limits=limits, quote_reader=lambda: quote(1.00, 1.20),
        sleep=no_sleep, clock=Ticker(),
    )
    prices = [r.limit_price for r in trading.requests]
    assert prices[0] == 2.00          # the original mid
    assert prices[-1] == 1.00         # the re-quoted bid


def test_a_failing_re_quote_keeps_the_original_ladder(config, limits):
    """Losing the quote feed must not abandon an exit mid-ladder."""
    trading = FakeTrading(script=[(OrderStatus.CANCELED, 0)] * 4)
    clients = FakeClients(config, trading)

    def boom():
        raise RuntimeError("feed down")

    place_exit(clients, symbol=SYMBOL, qty=1, quote=quote(1.90, 2.10),
               reason="target", limits=limits, quote_reader=boom,
               sleep=no_sleep, clock=Ticker())
    assert [r.limit_price for r in trading.requests][-1] == 1.90


# --- polling ---------------------------------------------------------------


def test_a_resting_order_is_cancelled_at_the_timeout(config):
    """An unattended resting order that fills later is a position nobody
    decided to open."""
    trading = FakeTrading(script=[(OrderStatus.NEW, 0)])
    clients = FakeClients(config, trading)
    request = build_single_leg(symbol=SYMBOL, qty=1, intent=Intent.BUY_TO_OPEN,
                               limit_price=2.00)
    order = trading.submit_order(request)
    final = poll_to_terminal(clients, str(order.id), timeout_seconds=3,
                             interval_seconds=1, sleep=no_sleep, clock=Ticker())
    assert str(order.id) in trading.cancelled
    assert final.status is OrderStatus.CANCELED


def test_a_filled_order_polls_once(config):
    trading = FakeTrading()
    clients = FakeClients(config, trading)
    request = build_single_leg(symbol=SYMBOL, qty=1, intent=Intent.BUY_TO_OPEN,
                               limit_price=2.00)
    order = trading.submit_order(request)
    assert is_filled(poll_to_terminal(clients, str(order.id), 10, 1,
                                      sleep=no_sleep, clock=Ticker()))
    assert trading.cancelled == []


def test_cancel_returns_broker_truth_not_the_call_result(config):
    """A cancel that returns is not a cancel that happened -- the order may
    have filled in the gap, and that position is real."""
    trading = FakeTrading()
    clients = FakeClients(config, trading)
    request = build_single_leg(symbol=SYMBOL, qty=1, intent=Intent.SELL_TO_CLOSE,
                               limit_price=2.00)
    order = trading.submit_order(request)
    order.status = OrderStatus.FILLED

    def already_filled(_):
        raise RuntimeError("order is already filled")

    trading.cancel_order_by_id = already_filled
    assert cancel(clients, str(order.id)).status is OrderStatus.FILLED


# --- no market orders, anywhere -------------------------------------------


def test_no_path_constructs_a_market_order(config, limits):
    """A bare market order on a 1-2% spread is a guaranteed cost with no upper
    bound. Asserted across both order paths rather than by inspection."""
    trading = FakeTrading(script=[(OrderStatus.CANCELED, 0)] * 6)
    clients = FakeClients(config, trading)
    place_entry(clients, symbol=SYMBOL, qty=1, quote=quote(), limits=limits,
                sleep=no_sleep, clock=Ticker())
    place_exit(clients, symbol=SYMBOL, qty=1, quote=quote(), reason="stop",
               limits=limits, sleep=no_sleep, clock=Ticker())
    assert trading.requests
    for request in trading.requests:
        assert request.limit_price is not None
        assert str(getattr(request, "type", "limit")).lower().endswith("limit")
