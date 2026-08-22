"""Alpaca auth, client construction, retry/backoff, and account reads.

Everything that talks to Alpaca goes through :func:`build_clients`, so the
paper guard, the mock guard and the retry policy exist in exactly one place.

Retry policy comes from ``config/limits.yaml`` (``broker.retry_attempts``,
``broker.backoff_base_seconds``). Per CLAUDE.md failure rule 4: exponential
backoff, N attempts, then the caller halts new entries rather than guessing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from alpaca.common.exceptions import APIError
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

from src.config import Config, ConfigError

log = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = [
    "AlpacaClients",
    "BrokerError",
    "MockModeError",
    "account_summary",
    "build_clients",
    "sizing_capital",
    "with_retry",
]


class BrokerError(RuntimeError):
    """A broker call failed after exhausting the configured retries."""


class MockModeError(RuntimeError):
    """A write was attempted while ``ALPACA_MOCK`` is set."""


@dataclass(frozen=True)
class AlpacaClients:
    """The three clients, plus the config they were built from."""

    trading: TradingClient
    options: OptionHistoricalDataClient
    stocks: StockHistoricalDataClient
    config: Config

    @property
    def options_feed(self) -> str:
        """From limits.yaml, never from .env -- one source for feeds."""
        return self.config.limits.get_str("broker.data_feed_options")

    @property
    def equities_feed(self) -> str:
        return self.config.limits.get_str("broker.data_feed_equities")

    @property
    def mock(self) -> bool:
        return self.config.env.mock

    def assert_writable(self, action: str) -> None:
        """Gate every order-placing path.

        ``ALPACA_MOCK`` exists so a session can be driven end to end without
        touching the broker. Reads stay live; writes raise. A mock flag that
        silently allowed orders through would be worse than no flag at all.
        """
        if self.mock:
            raise MockModeError(
                f"refusing to {action}: ALPACA_MOCK is set in .env. Unset it to place "
                "orders, or use --dry-run / the replay harness."
            )


def build_clients(config: Config) -> AlpacaClients:
    """Construct the Alpaca clients, refusing to arm against a live account.

    ``alpaca-py`` selects the trading endpoint with a ``paper`` boolean rather
    than a base URL, so the URL in ``.env`` is what we validate and then
    translate. ``ALPACA_DATA_URL`` overrides the market-data host, which is a
    separate endpoint from the trading one.
    """
    credentials = config.env.require_alpaca()

    require_paper = config.limits.get_bool("broker.require_paper_base_url")
    if require_paper and not credentials.is_paper:
        raise ConfigError(
            f"ALPACA_BASE_URL is {credentials.base_url!r}, which is not the paper "
            "endpoint, and limits.yaml sets broker.require_paper_base_url: true. "
            "Refusing to build a live trading client."
        )

    key, secret = credentials.api_key, credentials.secret_key
    data_url = credentials.data_url
    if data_url:
        log.info("market data host overridden by ALPACA_DATA_URL: %s", data_url)

    return AlpacaClients(
        trading=TradingClient(api_key=key, secret_key=secret, paper=credentials.is_paper),
        options=OptionHistoricalDataClient(api_key=key, secret_key=secret, url_override=data_url),
        stocks=StockHistoricalDataClient(api_key=key, secret_key=secret, url_override=data_url),
        config=config,
    )


# Alpaca returns these for genuinely transient conditions. Anything else -- a
# rejected order, a bad symbol, insufficient buying power -- is a real answer
# and must not be retried into a duplicate order.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, APIError):
        return getattr(exc, "status_code", None) in _RETRYABLE_STATUS
    # Connection resets, timeouts, DNS blips.
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


def with_retry(config: Config, description: str, call: Callable[[], T]) -> T:
    """Run ``call`` with exponential backoff on transient failures.

    Deliberately not a decorator: the call site names what it is doing, which
    is what ends up in the log and, later, in the decision log.
    """
    attempts = config.limits.get_int("broker.retry_attempts")
    base = config.limits.get_float("broker.backoff_base_seconds")

    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 -- re-raised below
            if not _is_retryable(exc):
                raise
            last = exc
            if attempt == attempts:
                break
            delay = base * (2 ** (attempt - 1))
            log.warning(
                "%s failed (attempt %d/%d): %s -- retrying in %.1fs",
                description, attempt, attempts, exc, delay,
            )
            time.sleep(delay)

    raise BrokerError(f"{description} failed after {attempts} attempts: {last}") from last


# ---------------------------------------------------------------------------
# Account reads
# ---------------------------------------------------------------------------

# `buying_power` on a margin account is equity x multiplier -- 4x on this paper
# account. Long options cannot be bought on margin, so sizing against it would
# overstate capacity fourfold. These fields are readable for reporting and are
# never an input to a size.
FORBIDDEN_SIZING_FIELDS = frozenset(
    {"buying_power", "daytrading_buying_power", "regt_buying_power", "multiplier"}
)


def sizing_capital(account: Any) -> float:
    """The only capital figure sizing may use: options buying power, else equity.

    Asserts the result cannot have come from a margin field. This is a
    deliberate belt-and-braces check on a mistake that is silent, plausible,
    and would inflate every position by the account multiplier.
    """
    equity = float(account.equity)
    raw = getattr(account, "options_buying_power", None)
    capital = float(raw) if raw is not None else equity
    source = "options_buying_power" if raw is not None else "equity"

    multiplier = float(getattr(account, "multiplier", 1) or 1)
    margin_bp = float(getattr(account, "buying_power", 0) or 0)

    if multiplier > 1 and margin_bp > equity and abs(capital - margin_bp) < 0.01:
        raise BrokerError(
            f"sizing capital resolved to margin buying_power ({margin_bp}) with "
            f"multiplier {multiplier} -- long options cannot use margin"
        )
    if capital > equity + 0.01:
        raise BrokerError(
            f"sizing capital {capital} from {source} exceeds equity {equity} -- refusing"
        )
    if capital <= 0:
        raise BrokerError(f"sizing capital from {source} is {capital}")

    log.debug("sizing capital %.2f from %s (equity %.2f)", capital, source, equity)
    return capital


def account_summary(clients: AlpacaClients) -> dict[str, Any]:
    """Equity, capital and options level. Read-only; safe to call anytime.

    ``buying_power`` is reported for visibility and labelled, so that nobody
    reads this dict and reaches for the wrong number.
    """
    account = with_retry(clients.config, "get_account", clients.trading.get_account)
    return {
        "account_number": account.account_number,
        "status": str(account.status),
        "equity": float(account.equity),
        "cash": float(account.cash),
        "options_buying_power": float(account.options_buying_power or 0),
        "sizing_capital": sizing_capital(account),
        "buying_power (margin, NOT for sizing)": float(account.buying_power),
        "multiplier": getattr(account, "multiplier", None),
        "options_approved_level": getattr(account, "options_approved_level", None),
        "options_trading_level": getattr(account, "options_trading_level", None),
        "trading_blocked": account.trading_blocked,
        "pattern_day_trader": account.pattern_day_trader,
    }
