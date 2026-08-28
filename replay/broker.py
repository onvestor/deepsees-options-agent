"""A stubbed broker: fills, positions, and the deterministic exits.

Orders never leave the process. That is the acceptance condition for Step 8 --
a full session must replay with no network access to the broker -- and it is
also what makes the harness safe to run against a filled-in ``config/`` and a
real ``.env`` without any chance of touching an account.

**Fills are deterministic, and that is a deliberate trade against realism.**
The 25 Aug fill study found mid limits fill or they do not: buys filled 55%,
sells 18%, six of eight fills landed under a second, and nothing filled between
11s and 60s. Modelling that faithfully means a coin flip, and a replay whose
result changes between runs cannot answer "did this prompt do better than that
one". So the model crosses a configured fraction of the half-spread every time.
At the default of 1.0 that means buying the ask and selling the bid, which is
the conservative reading and lines up with the measured round trip of roughly
one quoted spread.

**The deterministic exits live here, not in the harness.** Stop, target, max
hold and the expiry-week flat are armed on every position independently of any
model, exactly as they are in production, and :func:`deterministic_exit` is a
pure function of position state so it can be tested without a replay at all.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Literal

from src.brokers.alpaca.quotes import OptionQuote

log = logging.getLogger(__name__)

Side = Literal["buy", "sell"]
CONTRACT_MULTIPLIER = 100.0


class FillError(RuntimeError):
    """An order that cannot be filled, with the reason named."""


@dataclass(frozen=True)
class FillModel:
    """How far a marketable order crosses the quoted spread.

    ``cross_fraction`` of 1.0 pays the full half-spread -- the ask on a buy and
    the bid on a sell. Lower values model a limit that gets some price
    improvement; 0.0 models a mid fill, which the fill study says happens about
    half the time on buys and rarely on sells, so it is an optimistic bound
    rather than an expectation.
    """

    cross_fraction: float = 1.0

    def price(self, quote: OptionQuote, side: Side) -> float:
        mid, spread = quote.mid, quote.spread
        if mid is None or spread is None:
            raise FillError(
                f"{quote.symbol}: no two-sided quote, nothing to fill against"
            )
        half = (spread / 2.0) * self.cross_fraction
        return round(mid + half if side == "buy" else mid - half, 2)


@dataclass(frozen=True)
class Fill:
    contract_symbol: str
    side: Side
    qty: int
    price: float
    session: date

    @property
    def cash(self) -> float:
        """Signed cash effect. Negative on a buy."""
        notional = self.price * self.qty * CONTRACT_MULTIPLIER
        return -notional if self.side == "buy" else notional


@dataclass(frozen=True)
class Position:
    """One open position. Immutable; changes produce a new one."""

    contract_symbol: str
    symbol: str
    qty: int
    entry_premium: float
    entry_session: date
    expiry: date
    stop_pct: float
    target_pct: float
    structure: str = "single_leg"

    def premium_paid(self) -> float:
        return self.entry_premium * self.qty * CONTRACT_MULTIPLIER

    def pnl_pct(self, current_premium: float) -> float:
        if self.entry_premium <= 0:
            return 0.0
        return (current_premium - self.entry_premium) / self.entry_premium * 100.0

    def pnl_dollars(self, current_premium: float) -> float:
        return (current_premium - self.entry_premium) * self.qty * CONTRACT_MULTIPLIER


@dataclass(frozen=True)
class ClosedTrade:
    position: Position
    exit_premium: float
    exit_session: date
    reason: str

    @property
    def pnl(self) -> float:
        return self.position.pnl_dollars(self.exit_premium)

    @property
    def pnl_pct(self) -> float:
        return self.position.pnl_pct(self.exit_premium)

    @property
    def sessions_held(self) -> int:
        return (self.exit_session - self.position.entry_session).days

    @property
    def won(self) -> bool:
        return self.pnl > 0


@dataclass(frozen=True)
class ExitBounds:
    """The deterministic exits, read once from config.

    All four are armed on every position regardless of what Agent 5 says or
    fails to say. Agent 5 can tighten ``stop_pct``; it cannot reach any of the
    others.
    """

    stop_pct: float
    target_pct: float
    max_hold_sessions: int
    min_sessions_to_expiry: int

    @classmethod
    def from_limits(cls, limits: Any) -> "ExitBounds":
        return cls(
            stop_pct=limits.get_float("exits.stop_pct"),
            target_pct=limits.get_float("exits.target_pct"),
            max_hold_sessions=limits.get_int("exits.max_hold_sessions"),
            min_sessions_to_expiry=limits.get_int("exits.min_sessions_to_expiry"),
        )


def deterministic_exit(
    position: Position,
    current_premium: float,
    sessions_held: int,
    sessions_to_expiry: int,
    bounds: ExitBounds,
) -> str | None:
    """The exit reason, or None to keep holding. Pure.

    Order matters and is not arbitrary. Expiry week comes first because a
    position there must leave whatever its P&L says -- theta acceleration and
    widening spreads make the other tests unreliable exactly where they are
    most likely to fire. Stop before target, because a position that somehow
    satisfies both on the same mark is a gap, and a gap resolves against us.
    """
    if sessions_to_expiry <= bounds.min_sessions_to_expiry:
        return "expiry_week"
    pnl = position.pnl_pct(current_premium)
    if pnl <= position.stop_pct:
        return "stop"
    if pnl >= bounds.target_pct:
        return "target"
    if sessions_held >= bounds.max_hold_sessions:
        return "max_hold"
    return None


@dataclass
class StubBroker:
    """Positions and cash, in memory. Nothing here touches a network.

    Equity is start-of-replay cash plus realized P&L. Unrealized P&L is
    deliberately excluded from the sizing equity: the live path reads
    ``options_buying_power`` from the broker, and paper option positions do not
    contribute buying power the way an unrealized equity gain does. Sizing
    against marked-to-market premium would let a winning open position fund a
    larger next one, which is the opposite of how the caps are meant to work.
    """

    starting_equity: float
    fills: FillModel = field(default_factory=FillModel)
    positions: list[Position] = field(default_factory=list)
    closed: list[ClosedTrade] = field(default_factory=list)
    order_log: list[Fill] = field(default_factory=list)
    realized_pnl: float = 0.0

    # -- state -------------------------------------------------------------

    @property
    def equity(self) -> float:
        return self.starting_equity + self.realized_pnl

    @property
    def open_premium(self) -> float:
        return sum(p.premium_paid() for p in self.positions)

    def positions_in(self, symbol: str) -> int:
        return sum(1 for p in self.positions if p.symbol == symbol.upper())

    def position_for(self, contract_symbol: str) -> Position | None:
        return next(
            (p for p in self.positions if p.contract_symbol == contract_symbol), None
        )

    # -- orders ------------------------------------------------------------

    def buy_to_open(
        self,
        *,
        contract_symbol: str,
        symbol: str,
        qty: int,
        quote: OptionQuote,
        session: date,
        expiry: date,
        stop_pct: float,
        target_pct: float,
        structure: str = "single_leg",
    ) -> Position:
        """Open a position at the modelled fill price."""
        if qty <= 0:
            raise FillError(f"{contract_symbol}: refusing to buy {qty} contracts")
        price = self.fills.price(quote, "buy")
        if price <= 0:
            raise FillError(f"{contract_symbol}: modelled fill price {price}")

        fill = Fill(contract_symbol, "buy", qty, price, session)
        self.order_log.append(fill)
        position = Position(
            contract_symbol=contract_symbol,
            symbol=symbol.upper(),
            qty=qty,
            entry_premium=price,
            entry_session=session,
            expiry=expiry,
            stop_pct=stop_pct,
            target_pct=target_pct,
            structure=structure,
        )
        self.positions.append(position)
        log.info(
            "replay fill: bought %d %s at %.2f on %s",
            qty, contract_symbol, price, session,
        )
        return position

    def close(
        self, position: Position, quote: OptionQuote, session: date, reason: str
    ) -> ClosedTrade:
        """Close a whole position and book the realized P&L."""
        price = self.fills.price(quote, "sell")
        self.order_log.append(Fill(position.contract_symbol, "sell", position.qty, price, session))
        self.positions.remove(position)
        trade = ClosedTrade(position, price, session, reason)
        self.closed.append(trade)
        self.realized_pnl += trade.pnl
        log.info(
            "replay exit: %s %s at %.2f (%s) P&L %.2f",
            position.contract_symbol, reason, price, session, trade.pnl,
        )
        return trade

    def scale_out_half(
        self, position: Position, quote: OptionQuote, session: date
    ) -> ClosedTrade | None:
        """Sell half, keep the rest. Returns None when there is no half to sell.

        A single contract has no half. Agent 5 may legitimately ask for one
        anyway -- it is told the position size, but the schema does not stop it
        -- so this reports the impossibility rather than silently closing the
        whole thing, which would be a model output turning into a full exit.
        """
        half = position.qty // 2
        if half < 1:
            log.info(
                "replay: scale_out_half on %s ignored -- %d contract(s), no half",
                position.contract_symbol, position.qty,
            )
            return None

        price = self.fills.price(quote, "sell")
        self.order_log.append(Fill(position.contract_symbol, "sell", half, price, session))
        sold = replace(position, qty=half)
        trade = ClosedTrade(sold, price, session, "scale_out_half")
        self.closed.append(trade)
        self.realized_pnl += trade.pnl

        self.positions.remove(position)
        self.positions.append(replace(position, qty=position.qty - half))
        return trade

    def tighten_stop(self, position: Position, new_stop_pct: float) -> Position:
        """Move a stop, and only ever inward.

        The schema, the validator and Agent 5's own gate all check this
        already. It is checked once more at the point of application because a
        stop is the last protection on an open position and the cost of the
        redundant check is nothing.
        """
        if new_stop_pct <= position.stop_pct:
            raise FillError(
                f"{position.contract_symbol}: {new_stop_pct} does not tighten "
                f"{position.stop_pct}"
            )
        updated = replace(position, stop_pct=new_stop_pct)
        self.positions[self.positions.index(position)] = updated
        return updated

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        wins = [t for t in self.closed if t.won]
        return {
            "starting_equity": round(self.starting_equity, 2),
            "ending_equity": round(self.equity, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "trades": len(self.closed),
            "wins": len(wins),
            "losses": len(self.closed) - len(wins),
            "open_positions": len(self.positions),
            "orders": len(self.order_log),
            "exit_reasons": self._reason_counts(),
        }

    def _reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for trade in self.closed:
            counts[trade.reason] = counts.get(trade.reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
