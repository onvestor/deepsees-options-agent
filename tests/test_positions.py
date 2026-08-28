"""Position reads. The contract is that every one of them hits the broker.

The orchestrator's ``reconcile`` job is only worth running if it is
authoritative, so the load-bearing test here is not about parsing -- it is that
no call is served from anything but a fresh read. A cache in this module would
make reconciliation confirm our own bookkeeping instead of checking it.
"""
from __future__ import annotations

from datetime import date

import pytest

from src.brokers.alpaca.positions import (
    Book,
    OptionPosition,
    account_state,
    is_closed,
    is_option_symbol,
    read_position,
    reconcile,
)

CALL = "SPY261016C00500000"
PUT = "NVDA261016P00180000"


class RawPosition:
    """The shape Alpaca's positions endpoint returns.

    Note what is absent: strike, expiry and option type. They are parsed from
    the symbol, which is why an unparseable symbol is a position we cannot
    reason about.
    """

    def __init__(self, symbol, qty="1", avg=2.00, current=2.40, side="long"):
        self.symbol = symbol
        self.qty = qty
        self.avg_entry_price = str(avg)
        self.current_price = str(current)
        self.market_value = str(float(current) * 100 * abs(int(float(qty))))
        self.cost_basis = str(float(avg) * 100 * abs(int(float(qty))))
        self.unrealized_pl = str((float(current) - float(avg)) * 100 * abs(int(float(qty))))
        self.unrealized_plpc = "0.20"
        self.side = side


class FakeTrading:
    def __init__(self, positions=None, equity=100_000.0):
        self._positions = list(positions or [])
        self.calls = 0
        self.single_calls = 0
        self.equity = equity

    def get_all_positions(self):
        self.calls += 1
        return list(self._positions)

    def get_open_position(self, symbol):
        self.single_calls += 1
        for p in self._positions:
            if p.symbol == symbol:
                return p
        raise RuntimeError(f"position does not exist: {symbol}")

    def get_account(self):
        class A:
            equity = self.equity
            options_buying_power = self.equity
            cash = self.equity
            multiplier = "1"
        return A()


class FakeClients:
    def __init__(self, config, trading):
        self.config = config
        self.trading = trading


@pytest.fixture(scope="module")
def config():
    from src.config import load_config

    return load_config()


def clients_with(config, *positions, equity=100_000.0):
    return FakeClients(config, FakeTrading(positions, equity))


# --- the contract: always a live read --------------------------------------


def test_every_reconcile_hits_the_broker(config):
    """No caching, ever. A reconcile that returned what we already believed
    would manufacture confidence rather than check it."""
    clients = clients_with(config, RawPosition(CALL))
    for _ in range(3):
        reconcile(clients)
    assert clients.trading.calls == 3


def test_every_single_read_hits_the_broker(config):
    clients = clients_with(config, RawPosition(CALL))
    for _ in range(3):
        read_position(clients, CALL)
    assert clients.trading.single_calls == 3


def test_a_position_opened_behind_our_back_appears(config):
    """The cancellation race: a cancel that returns already-filled is a real
    position we never recorded locally."""
    trading = FakeTrading([])
    clients = FakeClients(config, trading)
    assert len(reconcile(clients)) == 0

    trading._positions.append(RawPosition(CALL))
    assert len(reconcile(clients)) == 1


def test_a_position_closed_behind_our_back_disappears(config):
    """Assignment, exercise and expiry happen with no order of ours."""
    trading = FakeTrading([RawPosition(CALL)])
    clients = FakeClients(config, trading)
    assert len(reconcile(clients)) == 1

    trading._positions.clear()
    assert len(reconcile(clients)) == 0


# --- parsing ---------------------------------------------------------------


def test_option_fields_come_from_the_symbol(config):
    [position] = reconcile(clients_with(config, RawPosition(CALL))).positions
    assert position.underlying == "SPY"
    assert position.expiry == date(2026, 10, 16)
    assert position.strike == 500.0
    assert position.option_type == "call"


def test_a_put_parses_as_a_put(config):
    [position] = reconcile(clients_with(config, RawPosition(PUT))).positions
    assert position.option_type == "put"
    assert position.underlying == "NVDA"


def test_equities_are_ignored_not_counted(config):
    """They must not count against option caps."""
    book = reconcile(clients_with(config, RawPosition(CALL), RawPosition("AAPL")))
    assert book.symbols == (CALL,)


def test_is_option_symbol_separates_the_two():
    assert is_option_symbol(CALL)
    assert not is_option_symbol("AAPL")


def test_an_unparseable_position_is_surfaced_not_dropped(config):
    """A position we cannot parse is one we cannot exit. Omitting it would
    make the book look smaller than it is."""

    class Broken(RawPosition):
        def __init__(self):
            super().__init__(CALL)
            self.avg_entry_price = "not-a-number"

    book = reconcile(clients_with(config, Broken()))
    assert len(book) == 0
    assert book.unparseable and book.unparseable[0][0] == CALL


def test_one_bad_row_does_not_hide_the_others(config):
    class Broken(RawPosition):
        def __init__(self):
            super().__init__(PUT)
            self.qty = "not-a-number"

    book = reconcile(clients_with(config, RawPosition(CALL), Broken()))
    assert book.symbols == (CALL,)
    assert len(book.unparseable) == 1


# --- book arithmetic -------------------------------------------------------


def test_open_premium_is_measured_at_entry_not_marked(config):
    """It feeds caps.max_open_premium. A cap on committed capital must not
    move because an open position gained."""
    book = reconcile(clients_with(config, RawPosition(CALL, qty="2", avg=2.00, current=9.0)))
    assert book.open_premium == pytest.approx(2.00 * 2 * 100)


def test_counts_per_underlying_feed_the_cap(config):
    book = reconcile(clients_with(
        config, RawPosition(CALL), RawPosition("SPY261016C00510000"), RawPosition(PUT)
    ))
    assert book.count_in("SPY") == 2
    assert book.count_in("NVDA") == 1
    assert book.count_in("TSLA") == 0


def test_pnl_is_a_percentage_of_premium_not_of_equity(config):
    """The unit every exit threshold is expressed in."""
    [position] = reconcile(clients_with(
        config, RawPosition(CALL, avg=2.00, current=3.00)
    )).positions
    assert position.pnl_pct() == pytest.approx(50.0)


def test_an_unmarked_position_has_no_pnl_rather_than_zero(config):
    """An unmarked position is not a flat one."""
    raw = RawPosition(CALL)
    raw.current_price = None
    [position] = reconcile(clients_with(config, raw)).positions
    assert position.current_price is None
    assert position.pnl_pct() is None


def test_the_book_stamps_when_it_was_read(config):
    book = reconcile(clients_with(config, RawPosition(CALL)))
    assert book.as_of is not None
    assert book.as_dict()["as_of"]


def test_lookup_by_symbol_is_case_insensitive(config):
    book = reconcile(clients_with(config, RawPosition(CALL)))
    assert book.get(CALL.lower()) is not None
    assert book.get("NOPE261016C00500000") is None


# --- absence is a normal answer -------------------------------------------


def test_a_missing_position_reads_as_none(config):
    """Alpaca raises. That is the common case after an exit fills."""
    assert read_position(clients_with(config), CALL) is None


def test_is_closed_reports_absence(config):
    assert is_closed(clients_with(config), CALL)
    assert not is_closed(clients_with(config, RawPosition(CALL)), CALL)


# --- the risk layer's view -------------------------------------------------


def test_account_state_is_built_from_broker_truth(config):
    clients = clients_with(config, RawPosition(CALL, qty="2", avg=2.00), equity=90_000.0)
    state = account_state(clients, underlying="SPY")
    assert state.equity == 90_000.0
    assert state.open_positions == 1
    assert state.positions_in_symbol == 1
    assert state.open_premium == pytest.approx(400.0)


def test_account_state_reads_positions_when_not_given_a_book(config):
    clients = clients_with(config, RawPosition(CALL))
    account_state(clients)
    assert clients.trading.calls == 1
