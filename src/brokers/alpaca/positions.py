"""Position reads. Always from Alpaca, never from local state.

**There is no cache in this module and there must never be one.** Every public
function performs a fresh read. That is a deliberate exception to the two-tier
caching everything else uses, and the reasons are specific:

* **The orchestrator's ``reconcile`` job is only worth running if it is
  authoritative.** A reconcile that returned what we already believed would
  confirm our own bookkeeping rather than check it, which is worse than not
  reconciling at all -- it manufactures confidence.

* **The cancellation race corrupts exactly this state.** A cancel that comes
  back "already filled" is a real position we did not know about. CLAUDE.md's
  entry-manager design requires headroom to be recomputed from reconciled
  broker state after every cancellation, precisely because local counters are
  what the race breaks.

* **Assignment, exercise and expiry happen without us.** An ITM contract
  auto-exercises at expiry, and Alpaca sells positions out within an hour of
  expiry when buying power is short. Both change the book with no order of
  ours involved. Local state cannot know; a read can.

**Paper NTAs lag by a day.** Exercise, assignment and expiry reach the
Activities endpoint the *next* day, though balances and positions update
instantly. So positions and equity here are current, and anything reconstructed
from activities is not -- do not build same-day reconciliation on them.

**A missing position is a normal answer, not an error.** Alpaca raises when
asked for a position that does not exist. That is the common case after an exit
fills, so it is translated into ``None`` rather than propagated.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable

from src.brokers.alpaca.client import AlpacaClients, with_retry
from src.options.occ import OccError
from src.options.occ import parse as parse_occ

log = logging.getLogger(__name__)

SHARES_PER_CONTRACT = 100


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class OptionPosition:
    """One open option position, as Alpaca reports it right now.

    The OCC fields are parsed from the symbol rather than taken from the API,
    because the positions endpoint does not return strike, expiry or type --
    and a position whose symbol will not parse is a position we cannot reason
    about, which :func:`reconcile` surfaces rather than drops.
    """

    symbol: str
    underlying: str
    qty: int
    avg_entry_price: float
    current_price: float | None
    market_value: float | None
    cost_basis: float | None
    unrealized_pl: float | None
    unrealized_plpc: float | None
    side: str
    expiry: date
    strike: float
    option_type: str

    @classmethod
    def from_api(cls, position: Any) -> "OptionPosition":
        symbol = str(position.symbol)
        parsed = parse_occ(symbol)
        qty = int(float(position.qty))
        return cls(
            symbol=symbol,
            underlying=parsed.root,
            qty=qty,
            avg_entry_price=float(position.avg_entry_price),
            current_price=_number(getattr(position, "current_price", None)),
            market_value=_number(getattr(position, "market_value", None)),
            cost_basis=_number(getattr(position, "cost_basis", None)),
            unrealized_pl=_number(getattr(position, "unrealized_pl", None)),
            unrealized_plpc=_number(getattr(position, "unrealized_plpc", None)),
            side=str(getattr(position, "side", "long")).split(".")[-1].lower(),
            expiry=parsed.expiry,
            strike=parsed.strike,
            option_type=parsed.option_type,
        )

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    @property
    def premium_paid(self) -> float:
        return self.avg_entry_price * abs(self.qty) * SHARES_PER_CONTRACT

    def pnl_pct(self) -> float | None:
        """P&L as a percentage of premium paid.

        The unit every exit threshold in this system is expressed in --
        ``exits.stop_pct`` and ``target_pct`` are percentages of premium, not
        of account equity. Returns ``None`` when the position has no mark,
        rather than defaulting to zero: an unmarked position is not a flat one.
        """
        if self.current_price is None or self.avg_entry_price <= 0:
            return None
        return (self.current_price - self.avg_entry_price) / self.avg_entry_price * 100.0

    def sessions_to_expiry(self, calendar: Any, session: date) -> int:
        return calendar.sessions_until(self.expiry, session)


@dataclass(frozen=True)
class Book:
    """Every option position at one moment, plus what could not be parsed.

    ``as_of`` is stamped at the read. It exists so a caller that holds a Book
    can tell how old its view is -- and so that holding one for longer than a
    single decision looks like the mistake it is.
    """

    positions: tuple[OptionPosition, ...]
    as_of: datetime
    unparseable: tuple[tuple[str, str], ...] = ()

    def __len__(self) -> int:
        return len(self.positions)

    def __iter__(self) -> Iterable[OptionPosition]:
        return iter(self.positions)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(p.symbol for p in self.positions)

    @property
    def underlyings(self) -> tuple[str, ...]:
        return tuple(sorted({p.underlying for p in self.positions}))

    @property
    def open_premium(self) -> float:
        """Total premium at risk, at entry prices.

        At entry prices rather than marked, because this feeds the caps
        (``caps.max_open_premium``), and a cap on committed capital should not
        move because an open position gained.
        """
        return sum(p.premium_paid for p in self.positions)

    @property
    def market_value(self) -> float:
        return sum(p.market_value or 0.0 for p in self.positions)

    @property
    def unrealized_pl(self) -> float:
        return sum(p.unrealized_pl or 0.0 for p in self.positions)

    def get(self, symbol: str) -> OptionPosition | None:
        target = symbol.strip().upper()
        return next((p for p in self.positions if p.symbol == target), None)

    def in_underlying(self, underlying: str) -> tuple[OptionPosition, ...]:
        target = underlying.strip().upper()
        return tuple(p for p in self.positions if p.underlying == target)

    def count_in(self, underlying: str) -> int:
        """Feeds ``caps.max_positions_per_symbol``, from broker truth."""
        return len(self.in_underlying(underlying))

    def as_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "positions": len(self.positions),
            "underlyings": list(self.underlyings),
            "open_premium": round(self.open_premium, 2),
            "market_value": round(self.market_value, 2),
            "unrealized_pl": round(self.unrealized_pl, 2),
            "unparseable": [{"symbol": s, "error": e} for s, e in self.unparseable],
        }


def is_option_symbol(symbol: str) -> bool:
    """Whether a position symbol is an option rather than an equity.

    ``get_all_positions`` returns both. Equities are not this system's concern
    and must not be counted against option caps -- but they are also not
    dropped silently anywhere a human might need to know about them, which is
    why :func:`reconcile` logs them.
    """
    try:
        parse_occ(symbol)
        return True
    except OccError:
        return False


def reconcile(clients: AlpacaClients) -> Book:
    """Read the whole book from Alpaca. Always a live call.

    This is the authoritative view. Nothing in this system may size, cap, or
    exit against a position set that did not come from here.
    """
    raw = with_retry(
        clients.config, "get_all_positions", clients.trading.get_all_positions
    )
    positions: list[OptionPosition] = []
    unparseable: list[tuple[str, str]] = []
    equities: list[str] = []

    for item in raw or []:
        symbol = str(getattr(item, "symbol", ""))
        if not is_option_symbol(symbol):
            equities.append(symbol)
            continue
        try:
            positions.append(OptionPosition.from_api(item))
        except Exception as exc:  # noqa: BLE001 -- a bad row must not hide the rest
            # Surfaced rather than dropped. A position we cannot parse is one
            # we cannot exit, and silently omitting it would make the book
            # look smaller than it is.
            unparseable.append((symbol, f"{type(exc).__name__}: {exc}"))
            log.error("position %s could not be parsed: %s", symbol, exc)

    if equities:
        log.info(
            "reconcile: ignoring %d non-option position(s): %s",
            len(equities), ", ".join(sorted(equities)),
        )
    log.info(
        "reconcile: %d option position(s), %.2f premium at risk",
        len(positions), sum(p.premium_paid for p in positions),
    )
    return Book(
        positions=tuple(sorted(positions, key=lambda p: p.symbol)),
        as_of=datetime.now(tz=timezone.utc),
        unparseable=tuple(unparseable),
    )


def read_position(clients: AlpacaClients, symbol: str) -> OptionPosition | None:
    """One position, or ``None`` if it is not open. Always a live call.

    Absence is the normal answer after an exit fills, so the broker's
    "position does not exist" error is translated rather than propagated.
    """
    target = symbol.strip().upper()
    try:
        raw = with_retry(
            clients.config,
            f"get_position({target})",
            lambda: clients.trading.get_open_position(target),
        )
    except Exception as exc:  # noqa: BLE001 -- absence is a normal answer
        log.info("no open position for %s (%s)", target, exc)
        return None
    if raw is None:
        return None
    return OptionPosition.from_api(raw)


def is_closed(clients: AlpacaClients, symbol: str) -> bool:
    """Whether a symbol has no open position. A live read, by definition."""
    return read_position(clients, symbol) is None


def account_state(
    clients: AlpacaClients,
    book: Book | None = None,
    *,
    entries_this_session: int = 0,
    underlying: str | None = None,
) -> Any:
    """Build the risk layer's :class:`AccountState` from broker truth.

    The sizing layer takes equity, options buying power, open premium and
    position counts. Every one of them comes from a live read here, so a
    position opened by a race -- or closed by an assignment we never saw --
    is reflected in the next size rather than in a surprise.
    """
    from src.brokers.alpaca.client import sizing_capital
    from src.risk.sizing import AccountState

    account = with_retry(clients.config, "get_account", clients.trading.get_account)
    book = book if book is not None else reconcile(clients)
    capital = sizing_capital(account)

    return AccountState(
        equity=float(account.equity),
        options_buying_power=capital,
        open_premium=book.open_premium,
        open_positions=len(book),
        positions_in_symbol=book.count_in(underlying) if underlying else 0,
        entries_this_session=entries_this_session,
        entries_this_symbol_this_session=0,
    )
