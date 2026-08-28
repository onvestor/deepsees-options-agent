"""Single-leg option order construction and submission.

**Debit structures only, one leg.** Verticals were designed, measured and not
shipped -- see "Debit verticals are unconstructible at swing DTE" in CLAUDE.md.
There is deliberately no ``mleg`` path here: an unused multi-leg builder would
be code nobody has ever seen work, sitting next to the one path that has.

Every constraint Alpaca imposes on an options order is enforced in code rather
than trusted to the caller, because each one fails at submission with a message
that does not name the field:

* ``qty`` must be a whole number, and ``notional`` must never be populated
* ``time_in_force`` must be ``day`` or ``gtc`` -- and we use ``day``, because a
  ``gtc`` entry can fill on a session whose thesis was formed days earlier
* ``extended_hours`` must be false or absent -- it is absent
* ``stop`` and ``stop_limit`` are single-leg only, and we use neither: there is
  no broker-side stop in this system at all, by design. Stops are ours.
* ``position_intent`` is set explicitly on every order

**Nothing here is a market order.** Not one path. A bare market order on an
option with a 1-2% spread is a guaranteed cost with no upper bound, and the
25 Aug fill study measured what the spread actually costs. Entries rest at the
mid; exits walk a ladder toward the bid.

**The two sides have different shapes, on purpose.**

*Entry is a single passive mid limit.* It fills or it does not. The fill study
found mid limits are bimodal rather than gradual -- six of eight fills landed
under a second and nothing filled between 11s and 60s -- so patience at a fixed
price buys almost nothing, and repricing is the entry order manager's job on
its own clock, not this module's.

*Exit is a stepped limit whose urgency is keyed to the exit reason.* A stop and
a profit target are not the same hurry. Every ladder still terminates at the
bid, so no exit can rest un-marketable forever; what urgency changes is how
fast it gets there.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Callable, Sequence

from alpaca.trading.enums import OrderSide, OrderStatus, PositionIntent, TimeInForce
from alpaca.trading.requests import LimitOrderRequest

from src.brokers.alpaca.client import AlpacaClients, BrokerError, sizing_capital, with_retry
from src.brokers.alpaca.quotes import OptionQuote

log = logging.getLogger(__name__)

SHARES_PER_CONTRACT = 100
"""One contract controls 100 shares. Contract spec, not a tunable."""

TERMINAL_STATUSES = frozenset(
    {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED}
)


class OrderError(BrokerError):
    """An order that must not be sent, with the violated constraint named."""


class Intent(str, Enum):
    """Which side of a position an order is on."""

    BUY_TO_OPEN = "buy_to_open"
    SELL_TO_CLOSE = "sell_to_close"

    @property
    def side(self) -> OrderSide:
        return OrderSide.BUY if self is Intent.BUY_TO_OPEN else OrderSide.SELL

    @property
    def position_intent(self) -> PositionIntent:
        return (
            PositionIntent.BUY_TO_OPEN
            if self is Intent.BUY_TO_OPEN
            else PositionIntent.SELL_TO_CLOSE
        )


class Urgency(str, Enum):
    """How fast an exit ladder walks to the bid."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# The mapping from *why* we are exiting to *how fast*. This is the whole point
# of keying urgency to the reason rather than to elapsed time: a stop is
# running risk and a target is banking profit, and a ladder that treated them
# identically would either give back gains or sit through a loss.
EXIT_URGENCY: dict[str, Urgency] = {
    "stop": Urgency.HIGH,
    "expiry_week": Urgency.HIGH,
    "agent_exit_now": Urgency.HIGH,
    "killswitch": Urgency.HIGH,
    "max_hold": Urgency.MEDIUM,
    "reconcile_orphan": Urgency.MEDIUM,
    "target": Urgency.LOW,
    "scale_out_half": Urgency.LOW,
}

DEFAULT_EXIT_URGENCY = Urgency.HIGH
"""An unrecognised reason exits fast.

Failing closed in the direction of *leaving*: an exit reason this module has
not been taught is one nobody has reasoned about, and holding a position
through it because the ladder was patient is the worse error.
"""


def urgency_for(reason: str) -> Urgency:
    return EXIT_URGENCY.get(reason, DEFAULT_EXIT_URGENCY)


# --- price arithmetic ------------------------------------------------------


def round_penny(value: float) -> float:
    """Round to a cent, away from float error.

    Options quote in pennies below $3 and nickels above on many venues, but
    Alpaca accepts penny increments on the paper API and rejects sub-penny.
    The nickel rule is not modelled -- a rejected order names the field, and
    guessing the increment per contract would be a silent mispricing.
    """
    return round(value + 1e-9, 2)


def mid_price(quote: OptionQuote) -> float:
    mid = quote.mid
    if mid is None:
        raise OrderError(
            f"{quote.symbol}: no two-sided quote, so there is no mid to rest at"
        )
    return round_penny(mid)


def step_ladder(quote: OptionQuote, steps: int) -> list[float]:
    """Limit prices walking from the mid to the bid, inclusive of both.

    ``steps`` is the number of prices, not the number of moves. One step is
    the bid outright -- the fully marketable case, used when there is no time
    to be clever.

    The ladder always ends at the bid. A ladder that stopped short would leave
    an exit resting at a price nobody is obliged to pay, which is the failure
    mode the deterministic exits exist to prevent.
    """
    if steps < 1:
        raise OrderError(f"a ladder needs at least one step, got {steps}")
    mid, bid = quote.mid, quote.bid
    if mid is None or bid is None:
        raise OrderError(f"{quote.symbol}: no two-sided quote to build a ladder from")
    if steps == 1:
        return [round_penny(bid)]

    span = mid - bid
    return [
        round_penny(mid - span * (i / (steps - 1)))
        for i in range(steps)
    ]


# --- configuration ---------------------------------------------------------


@dataclass(frozen=True)
class ExecutionLimits:
    """The execution policy, read once.

    Lives under ``execution:`` beside the entry order manager's own keys. The
    manager's remain unset and unbuilt; these are the ones the order path needs
    today, and they are separate keys precisely so that tuning the exit ladder
    cannot move the entry repricing cadence.
    """

    entry_fill_timeout_seconds: int
    exit_step_seconds: dict[Urgency, int]
    exit_steps: dict[Urgency, int]
    poll_interval_seconds: int

    @classmethod
    def from_limits(cls, limits: Any) -> "ExecutionLimits":
        return cls(
            entry_fill_timeout_seconds=limits.get_int(
                "execution.entry_fill_timeout_seconds"
            ),
            exit_step_seconds={
                u: limits.get_int(f"execution.exit_step_seconds_{u.value}")
                for u in Urgency
            },
            exit_steps={
                u: limits.get_int(f"execution.exit_steps_{u.value}") for u in Urgency
            },
            poll_interval_seconds=limits.get_int("exits.poll_interval_seconds"),
        )


# --- request construction --------------------------------------------------


def build_single_leg(
    *,
    symbol: str,
    qty: int,
    intent: Intent,
    limit_price: float,
) -> LimitOrderRequest:
    """A validated single-leg option limit order.

    Every check here corresponds to a documented Alpaca constraint, and each
    raises naming the field rather than letting the API reject it with
    something less specific.
    """
    if not symbol or not symbol.strip():
        raise OrderError("symbol is required")
    if isinstance(qty, bool) or not isinstance(qty, int):
        raise OrderError(
            f"qty must be a whole number of contracts, got {type(qty).__name__} "
            f"{qty!r} -- Alpaca rejects fractional option quantities"
        )
    if qty <= 0:
        raise OrderError(f"qty must be positive, got {qty}")
    if limit_price <= 0:
        raise OrderError(f"limit_price must be positive, got {limit_price}")
    if round_penny(limit_price) != limit_price:
        raise OrderError(
            f"limit_price {limit_price} is sub-penny; round it before submitting"
        )

    # notional is never passed. extended_hours is never passed. Both are
    # omissions the API treats as correct, and both would be rejections if set.
    return LimitOrderRequest(
        symbol=symbol.strip().upper(),
        qty=qty,
        side=intent.side,
        type="limit",
        time_in_force=TimeInForce.DAY,
        limit_price=limit_price,
        position_intent=intent.position_intent,
    )


def assert_affordable(clients: AlpacaClients, limit_price: float, qty: int) -> float:
    """Premium must fit inside sizing capital -- never margin buying power.

    ``sizing_capital`` is the guard that exists because a 4x margin multiplier
    makes ``buying_power`` look like four times the money that can actually be
    committed to long premium.
    """
    account = with_retry(clients.config, "get_account", clients.trading.get_account)
    capital = sizing_capital(account)
    premium = limit_price * SHARES_PER_CONTRACT * qty
    if premium > capital:
        raise OrderError(
            f"premium {premium:.2f} exceeds sizing capital {capital:.2f} "
            "(options_buying_power/equity, not margin buying power)"
        )
    return premium


# --- submission and polling ------------------------------------------------


def submit(clients: AlpacaClients, request: LimitOrderRequest, description: str) -> Any:
    """Send one order. Guarded by mock mode, retried on transport errors."""
    clients.assert_writable(description)
    order = with_retry(
        clients.config, description, lambda: clients.trading.submit_order(request)
    )
    log.info(
        "%s submitted: id=%s %s %s qty=%s limit=%s",
        description, order.id, order.side, order.symbol, order.qty, order.limit_price,
    )
    return order


def get_order(clients: AlpacaClients, order_id: str) -> Any:
    return with_retry(
        clients.config, "get_order", lambda: clients.trading.get_order_by_id(order_id)
    )


def cancel(clients: AlpacaClients, order_id: str) -> Any:
    """Cancel and poll to a terminal status.

    **A cancel that returns is not a cancel that happened.** The order may fill
    in the gap, and the resulting position is real. Callers must re-read
    broker state afterwards rather than assuming the cancel succeeded -- see
    the cancellation race in CLAUDE.md's entry-manager design.
    """
    clients.assert_writable("cancel_order")
    try:
        with_retry(
            clients.config, "cancel_order",
            lambda: clients.trading.cancel_order_by_id(order_id),
        )
    except Exception as exc:  # noqa: BLE001 -- an already-terminal order is normal
        log.info("cancel %s returned %s; polling for the truth", order_id, exc)
    return get_order(clients, order_id)


def poll_to_terminal(
    clients: AlpacaClients,
    order_id: str,
    timeout_seconds: int,
    interval_seconds: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Any:
    """Poll until the order is terminal or the timeout expires.

    On timeout the order is **cancelled rather than left resting**. An
    unattended resting order that fills later is a position nobody decided to
    open, and the caller has already moved on.

    ``sleep`` and ``clock`` are injected so the timeout path is testable
    without spending the timeout.
    """
    interval = interval_seconds or clients.config.limits.get_int(
        "exits.poll_interval_seconds"
    )
    deadline = clock() + timeout_seconds
    order = get_order(clients, order_id)
    while order.status not in TERMINAL_STATUSES and clock() < deadline:
        sleep(max(0.0, min(interval, deadline - clock())))
        order = get_order(clients, order_id)

    if order.status not in TERMINAL_STATUSES:
        log.warning(
            "order %s still %s after %ds -- cancelling rather than leaving it to rest",
            order_id, order.status, timeout_seconds,
        )
        order = cancel(clients, order_id)
    return order


def filled_qty(order: Any) -> int:
    return int(float(getattr(order, "filled_qty", 0) or 0))


def is_filled(order: Any) -> bool:
    return getattr(order, "status", None) == OrderStatus.FILLED


# --- the two paths ---------------------------------------------------------


@dataclass(frozen=True)
class OrderResult:
    """What an order attempt did, and at what price.

    **Fills accumulate across rungs.** A stepped exit can fill one contract at
    the mid and two at the bid, and reading only the last order would report
    the position as still open. ``fills`` carries every (qty, price) pair so
    both the total and a true average survive.
    """

    order: Any
    intent: Intent
    requested_qty: int
    limit_prices: tuple[float, ...]
    attempts: int = 1
    fills: tuple[tuple[int, float], ...] = ()

    @property
    def status(self) -> Any:
        return getattr(self.order, "status", None)

    @property
    def filled(self) -> int:
        """Contracts filled across every rung, not just the last one."""
        if self.fills:
            return sum(qty for qty, _ in self.fills)
        return filled_qty(self.order)

    @property
    def complete(self) -> bool:
        return self.filled >= self.requested_qty

    @property
    def fill_price(self) -> float | None:
        """Volume-weighted average across rungs.

        The last rung's price is not the trade's price when a ladder filled in
        pieces, and the realised round-trip cost is measured from this number.
        """
        if self.fills:
            total = sum(qty for qty, _ in self.fills)
            if total <= 0:
                return None
            return round(sum(qty * price for qty, price in self.fills) / total, 4)
        price = getattr(self.order, "filled_avg_price", None)
        return float(price) if price is not None else None

    @property
    def order_id(self) -> str:
        return str(getattr(self.order, "id", ""))


def place_entry(
    clients: AlpacaClients,
    *,
    symbol: str,
    qty: int,
    quote: OptionQuote,
    limits: ExecutionLimits | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> OrderResult:
    """A passive mid limit. One price, one attempt.

    Deliberately does not chase. The fill study says a mid limit fills almost
    immediately or not at all, so stepping an *entry* belongs to the entry
    order manager on its own cadence -- and that manager derives its step
    ceiling per contract from whether the trade is still worth taking at the
    higher price, which is a judgment this module has no inputs for.
    """
    limits = limits or ExecutionLimits.from_limits(clients.config.limits)
    price = mid_price(quote)
    assert_affordable(clients, price, qty)

    request = build_single_leg(
        symbol=symbol, qty=qty, intent=Intent.BUY_TO_OPEN, limit_price=price
    )
    order = submit(clients, request, "buy_to_open")
    order = poll_to_terminal(
        clients, str(order.id), limits.entry_fill_timeout_seconds,
        limits.poll_interval_seconds, sleep=sleep, clock=clock,
    )
    got = filled_qty(order)
    fills = ((got, float(order.filled_avg_price)),) if got and order.filled_avg_price else ()
    return OrderResult(order, Intent.BUY_TO_OPEN, qty, (price,), fills=fills)


def place_exit(
    clients: AlpacaClients,
    *,
    symbol: str,
    qty: int,
    quote: OptionQuote,
    reason: str,
    quote_reader: Callable[[], OptionQuote] | None = None,
    limits: ExecutionLimits | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> OrderResult:
    """A stepped limit walking from the mid to the bid, paced by the reason.

    Each step cancels the previous order and re-places lower. The ladder always
    ends at the bid, so a position that must leave will leave.

    ``quote_reader`` re-reads the quote between steps when supplied. Without it
    the ladder is computed once from ``quote``, which is correct for a fast
    ladder and stale for a slow one -- so a caller running a LOW-urgency exit
    should supply it.

    **Partial fills are honoured.** Each step re-places only the unfilled
    remainder. Re-placing the full quantity would over-sell a position that
    partially filled on the previous rung, which on a long option means going
    short -- an outcome this system has no path to and no approval for.
    """
    limits = limits or ExecutionLimits.from_limits(clients.config.limits)
    urgency = urgency_for(reason)
    steps = limits.exit_steps[urgency]
    pause = limits.exit_step_seconds[urgency]

    prices = step_ladder(quote, steps)
    log.info(
        "exit %s (%s, urgency=%s): ladder %s over %d step(s), %ds apart",
        symbol, reason, urgency.value, prices, steps, pause,
    )

    remaining = qty
    used: list[float] = []
    fills: list[tuple[int, float]] = []
    order: Any = None
    attempts = 0

    # Indexed rather than `for ... in enumerate(prices)`: the re-quote below
    # rebinds `prices`, and an already-started iterator would keep walking the
    # stale ladder while appearing to have been updated.
    index = 0
    while index < len(prices) and remaining > 0:
        price = prices[index]
        attempts += 1
        used.append(price)
        request = build_single_leg(
            symbol=symbol, qty=remaining, intent=Intent.SELL_TO_CLOSE, limit_price=price
        )
        order = submit(clients, request, f"sell_to_close[{reason}:{index + 1}]")
        order = poll_to_terminal(
            clients, str(order.id), pause, limits.poll_interval_seconds,
            sleep=sleep, clock=clock,
        )
        got = filled_qty(order)
        if got:
            fills.append((got, float(order.filled_avg_price or price)))
            remaining -= got
        if remaining <= 0:
            break

        if index < len(prices) - 1 and quote_reader is not None:
            # A slow ladder against a stale quote walks to a bid that moved.
            try:
                prices = _requote(quote_reader(), prices, index, steps)
            except Exception as exc:  # noqa: BLE001 -- keep the original ladder
                log.warning("re-quote failed (%s); continuing on the original ladder", exc)
        index += 1

    if remaining > 0:
        log.error(
            "exit %s (%s): ladder exhausted with %d of %d contract(s) unfilled at "
            "the bid -- the position is still open and needs attention",
            symbol, reason, remaining, qty,
        )
    return OrderResult(
        order, Intent.SELL_TO_CLOSE, qty, tuple(used), attempts, tuple(fills)
    )


def _requote(quote: OptionQuote, current: list[float], index: int, steps: int) -> list[float]:
    """Rebuild the remaining rungs against a fresh quote.

    The rungs already used are kept so the ladder's history stays honest, and
    only the future ones move.
    """
    fresh = step_ladder(quote, steps)
    return current[: index + 1] + fresh[index + 1 :]


def open_orders(clients: AlpacaClients, symbol: str | None = None) -> list[Any]:
    """Live orders, read from the broker. Never from local state."""
    from alpaca.trading.requests import GetOrdersRequest

    request = GetOrdersRequest(status="open", symbols=[symbol] if symbol else None)
    return list(
        with_retry(
            clients.config, "get_orders", lambda: clients.trading.get_orders(request)
        )
        or []
    )
