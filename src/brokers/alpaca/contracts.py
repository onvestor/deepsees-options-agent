"""Paginated option contract discovery.

``GET /v2/options/contracts`` returns strike, expiry, type, style,
``open_interest`` and ``close_price``. **No bid/ask and no greeks** -- those
need the snapshots endpoint, which is :mod:`src.brokers.alpaca.quotes`. The
prefilter's spread and delta tests depend entirely on that second call.

Two documented traps, both closed here rather than left to call sites:

* **The default ``expiration_date_lte`` is next weekend.** A caller that omits
  date bounds silently gets a few days of chain and no error. :func:`fetch`
  therefore *requires* both bounds and raises if either is missing -- there is
  no code path that can issue an unbounded query.
* **Results are paginated** via ``page_token`` with a default ``limit`` of
  100. A single page is not the chain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Literal

from alpaca.trading.enums import ContractType
from alpaca.trading.requests import GetOptionContractsRequest

from src.brokers.alpaca.cache import MarketDataCache
from src.brokers.alpaca.client import AlpacaClients, BrokerError, with_retry
from src.options.occ import OccError, parse as parse_occ

log = logging.getLogger(__name__)

__all__ = ["ContractSpec", "fetch"]

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class ContractSpec:
    """A contract as Alpaca describes it, cross-checked against its own symbol.

    ``from_api`` parses the OCC symbol and asserts the parsed strike, expiry
    and type match the fields Alpaca returned separately. They always have so
    far -- 320 recorded contracts round-trip exactly -- and the day they stop,
    this raises rather than silently sizing a position against the wrong
    strike.
    """

    symbol: str
    underlying: str
    root: str
    expiry: date
    strike: float
    option_type: OptionType
    style: str
    open_interest: int
    open_interest_date: date | None
    close_price: float | None
    close_price_date: date | None
    size: int
    tradable: bool
    status: str

    @classmethod
    def from_api(cls, contract: Any) -> "ContractSpec":
        symbol = contract.symbol
        try:
            parsed = parse_occ(symbol)
        except OccError as exc:
            raise BrokerError(f"Alpaca returned an unparseable OCC symbol {symbol!r}: {exc}") from exc

        strike = float(contract.strike_price)
        option_type = _as_str(contract.type)
        expiry = contract.expiration_date

        mismatches = []
        if abs(parsed.strike - strike) > 1e-9:
            mismatches.append(f"strike {parsed.strike} != {strike}")
        if parsed.expiry != expiry:
            mismatches.append(f"expiry {parsed.expiry} != {expiry}")
        if parsed.option_type != option_type:
            mismatches.append(f"type {parsed.option_type} != {option_type}")
        if mismatches:
            raise BrokerError(f"{symbol}: symbol disagrees with fields -- {'; '.join(mismatches)}")

        return cls(
            symbol=symbol,
            underlying=contract.underlying_symbol,
            root=contract.root_symbol or parsed.root,
            expiry=expiry,
            strike=strike,
            option_type=option_type,  # type: ignore[arg-type]
            style=_as_str(contract.style),
            open_interest=int(contract.open_interest or 0),
            open_interest_date=getattr(contract, "open_interest_date", None),
            close_price=float(contract.close_price) if contract.close_price is not None else None,
            close_price_date=getattr(contract, "close_price_date", None),
            size=int(contract.size or 100),
            tradable=bool(getattr(contract, "tradable", True)),
            status=_as_str(getattr(contract, "status", "active")),
        )

    @property
    def is_call(self) -> bool:
        return self.option_type == "call"


def _as_str(value: Any) -> str:
    """Alpaca returns enums in some paths and plain strings in others."""
    return str(getattr(value, "value", value))


def _cache_key(
    underlying: str,
    expiry_gte: date,
    expiry_lte: date,
    option_type: OptionType | None,
    strike_gte: float | None,
    strike_lte: float | None,
) -> tuple:
    return ("contracts", underlying, expiry_gte, expiry_lte, option_type, strike_gte, strike_lte)


def fetch(
    clients: AlpacaClients,
    underlying: str,
    expiry_gte: date,
    expiry_lte: date,
    option_type: OptionType | None = None,
    strike_gte: float | None = None,
    strike_lte: float | None = None,
    cache: MarketDataCache | None = None,
) -> list[ContractSpec]:
    """Fetch every contract matching the bounds, following pagination.

    ``expiry_gte`` and ``expiry_lte`` are mandatory. Alpaca's default upper
    bound is next weekend, so an omitted bound is not "no filter" -- it is a
    silently truncated chain, which is worse than an error.
    """
    if expiry_gte is None or expiry_lte is None:
        raise ValueError(
            "expiry_gte and expiry_lte are required -- Alpaca defaults "
            "expiration_date_lte to next weekend, so an unbounded query silently "
            "returns a truncated chain"
        )
    if expiry_lte < expiry_gte:
        raise ValueError(f"expiry_lte {expiry_lte} is before expiry_gte {expiry_gte}")

    underlying = underlying.strip().upper()
    key = _cache_key(underlying, expiry_gte, expiry_lte, option_type, strike_gte, strike_lte)

    if cache is not None:
        hit, cached = cache.contracts.get(key)
        if hit:
            log.debug("contracts cache hit: %s (%d contracts)", underlying, len(cached))
            return list(cached)

    config = clients.config
    page_limit = config.limits.get_int("broker.contracts_page_limit")
    max_pages = config.limits.get_int("broker.max_contract_pages")

    collected: list[ContractSpec] = []
    page_token: str | None = None
    seen_tokens: set[str] = set()
    pages = 0

    while True:
        request = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            status="active",
            expiration_date_gte=expiry_gte,
            expiration_date_lte=expiry_lte,
            strike_price_gte=str(strike_gte) if strike_gte is not None else None,
            strike_price_lte=str(strike_lte) if strike_lte is not None else None,
            type=ContractType(option_type) if option_type else None,
            limit=page_limit,
            page_token=page_token,
        )
        page = with_retry(
            config,
            f"get_option_contracts({underlying} p{pages + 1})",
            lambda r=request: clients.trading.get_option_contracts(r),
        )
        collected.extend(ContractSpec.from_api(c) for c in (page.option_contracts or []))
        pages += 1

        page_token = getattr(page, "next_page_token", None)
        if not page_token:
            break
        if page_token in seen_tokens:
            raise BrokerError(
                f"pagination loop: page_token {page_token!r} repeated for {underlying}"
            )
        seen_tokens.add(page_token)
        if pages >= max_pages:
            raise BrokerError(
                f"{underlying}: exceeded broker.max_contract_pages ({max_pages}) with a "
                "next_page_token still set -- narrow the date or strike bounds"
            )

    log.info(
        "%s: %d contracts over %d page(s), expiry %s..%s",
        underlying, len(collected), pages, expiry_gte, expiry_lte,
    )
    if cache is not None:
        cache.contracts.put(key, list(collected))
    return collected


def group_by_expiry(contracts: Iterable[ContractSpec]) -> dict[date, list[ContractSpec]]:
    grouped: dict[date, list[ContractSpec]] = {}
    for contract in contracts:
        grouped.setdefault(contract.expiry, []).append(contract)
    return {expiry: sorted(items, key=lambda c: c.strike) for expiry, items in sorted(grouped.items())}
