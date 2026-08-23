"""Earnings dates from Financial Modeling Prep, with a fail-closed cache.

Alpaca does not supply earnings dates. Its corporate-announcements endpoint
accepts only ``dividend``, ``merger``, ``spinoff`` and ``split`` -- verified
against the live API, not assumed. So this is a second provider, and every
second provider is a new way for the system to be quietly wrong.

**Use a symbol-scoped endpoint.** ``/stable/earnings-calendar`` accepts a
``symbol`` parameter and silently ignores it -- the payload is byte-identical
with and without it -- then the free tier caps the response to a middle slice
of the requested range. Filtering that client-side yields "the earliest row for
this symbol that happened to survive the cap", which is not the symbol's next
earnings date and is indistinguishable from one. Measured 2026-08-23: NVDA
reported in three days and the parse returned no date at all. It failed closed
that time; the same bug with a later row inside the slice reads as *clear* and
buys straight into a print. ``/stable/earnings?symbol=X`` is the scoped call.

**Fail closed, in four distinct ways.** An earnings exclusion that degrades to
"no date known, trade freely" is worse than no exclusion at all, because it
looks like it is working:

1. **Unknown** -- no entry for the symbol, or an entry with no date -> excluded.
2. **Stale** -- the cache is older than ``earnings.max_cache_age_hours`` ->
   excluded. A date fetched last week is not evidence about this week.
3. **Fetch failure** -- the refresh raises rather than returning an empty map,
   so a provider outage cannot silently empty the exclusion list.
4. **Unresolved at startup** -- :func:`assert_universe_resolves` refuses to run
   a session in which any symbol resolves to neither a real date nor an
   explicit no-earnings declaration. The three rules above make a broken feed
   *safe*; this one makes it *visible*, which is the part that was missing.

**The no-earnings class is declared, not inferred.** An index ETF has no print,
so rule 1 would block SPY and QQQ forever behind an exclusion that looks
healthy. ``config/universe.yaml: no_earnings`` names those instruments
explicitly. It is a claim about the instrument, never a way to skip the check:
a symbol in the class that *does* return a date is a contradiction and is
excluded loudly.

Each cached entry carries ``date``, ``confirmed`` and ``fetched_at``.
``confirmed`` matters: an *estimated* earnings date and a company-confirmed one
are different facts, and treating an estimate as authoritative is the quiet
way this exclusion fails. Configuration decides whether unconfirmed counts as
unknown.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import urlencode

import requests

log = logging.getLogger(__name__)

__all__ = [
    "EarningsCalendar",
    "EarningsEntry",
    "EarningsError",
    "EarningsVerdict",
    "assert_universe_resolves",
    "evaluate_exclusion",
]


class EarningsError(RuntimeError):
    """The earnings provider could not be reached or returned nonsense."""


@dataclass(frozen=True)
class EarningsEntry:
    """One symbol's next earnings date, and how much we trust it."""

    symbol: str
    date: str | None                 # ISO date, or None if the provider had none
    confirmed: bool | None           # None = the payload carried no signal either way
    fetched_at: str                  # ISO8601 UTC
    source: str = "fmp"

    @property
    def as_date(self) -> date | None:
        return date.fromisoformat(self.date) if self.date else None

    @property
    def fetched_datetime(self) -> datetime:
        stamp = datetime.fromisoformat(self.fetched_at.replace("Z", "+00:00"))
        return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)

    def age_hours(self, now: datetime) -> float:
        return (now - self.fetched_datetime).total_seconds() / 3600.0

    def is_stale(self, now: datetime, max_age_hours: float) -> bool:
        return self.age_hours(now) > max_age_hours

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EarningsVerdict:
    """Whether a symbol is tradeable, and why not if it isn't."""

    symbol: str
    excluded: bool
    reason: str
    earnings_date: str | None = None
    confirmed: bool | None = None
    sessions_until: int | None = None
    age_hours: float | None = None
    no_earnings_class: bool = False   # resolved by declaration, not by a date

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EarningsCalendar:
    """Fetches, caches and serves next-earnings dates.

    The cache is a plain JSON file, refetchable at any time and gitignored --
    it is provider data, not a decision record.
    """

    def __init__(
        self,
        api_key: str | None,
        cache_path: Path,
        base_url: str,
        path: str,
        timeout: float,
        max_rows: int,
        clock: Callable[[], datetime] | None = None,
        session: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.cache_path = Path(cache_path)
        self.base_url = base_url.rstrip("/")
        self.path = "/" + path.strip("/")
        self.timeout = timeout
        self.max_rows = max_rows
        self._clock = clock or (lambda: datetime.now(tz=timezone.utc))
        self._session = session or requests
        self._entries: dict[str, EarningsEntry] = {}
        self.load()

    # -- config adapter ------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> "EarningsCalendar":
        limits = config.limits
        return cls(
            api_key=config.env.fmp_api_key,
            cache_path=config.ensure_cache_dir() / "earnings.json",
            base_url=limits.get_str("earnings.base_url"),
            path=limits.get_str("earnings.path"),
            timeout=limits.get_float("earnings.request_timeout_seconds"),
            max_rows=limits.get_int("earnings.max_rows"),
            **kwargs,
        )

    # -- cache ---------------------------------------------------------------

    def load(self) -> None:
        if not self.cache_path.is_file():
            return
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # A corrupt cache means "we know nothing", which fails closed.
            log.warning("earnings cache unreadable (%s) -- treating as empty", exc)
            return
        for symbol, row in (raw.get("entries") or {}).items():
            try:
                self._entries[symbol.upper()] = EarningsEntry(**row)
            except TypeError:
                log.warning("discarding malformed earnings cache entry for %s", symbol)

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_note": "Provider cache. Refetchable, gitignored, not a decision record.",
            "saved_at": self._clock().isoformat(),
            "entries": {s: e.to_dict() for s, e in sorted(self._entries.items())},
        }
        self.cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get(self, symbol: str) -> EarningsEntry | None:
        return self._entries.get(symbol.strip().upper())

    def __len__(self) -> int:
        return len(self._entries)

    # -- fetching ------------------------------------------------------------

    def refresh(self, symbols: Sequence[str]) -> dict[str, EarningsEntry]:
        """Refetch every symbol. Raises on failure -- never returns a partial map.

        Returning what succeeded would empty the exclusion list for whatever
        failed, which is the precise failure this module exists to prevent.
        """
        if not self.api_key:
            raise EarningsError(
                "FMP_API_KEY is not set -- the earnings exclusion cannot run, and "
                "running without it would trade through earnings prints"
            )

        fetched_at = self._clock().isoformat()
        results: dict[str, EarningsEntry] = {}
        failures: list[str] = []

        for symbol in dict.fromkeys(s.strip().upper() for s in symbols if s):
            try:
                results[symbol] = self._fetch_one(symbol, fetched_at)
            except Exception as exc:  # noqa: BLE001 -- aggregated below
                log.warning("earnings fetch failed for %s: %s", symbol, exc)
                failures.append(f"{symbol}: {exc}")

        if failures:
            raise EarningsError(
                f"earnings refresh failed for {len(failures)} symbol(s): "
                + "; ".join(failures[:5])
            )

        self._entries.update(results)
        self.save()
        log.info("earnings refreshed for %d symbol(s)", len(results))
        return results

    def _fetch_one(self, symbol: str, fetched_at: str) -> EarningsEntry:
        """One symbol, from the symbol-scoped endpoint.

        No date bounds: this endpoint returns that symbol's most recent
        ``max_rows`` reports, past and future, and we pick the earliest one at
        or after today. Asking for a date range here is what broke the previous
        implementation -- the range parameters belong to the market-wide
        calendar path, which ignores ``symbol`` entirely.
        """
        today = self._clock().date()
        params = {"symbol": symbol, "limit": self.max_rows, "apikey": self.api_key}
        url = f"{self.base_url}{self.path}?{urlencode(params)}"
        response = self._session.get(url, timeout=self.timeout)
        if response.status_code != 200:
            raise EarningsError(f"HTTP {response.status_code}")
        rows = response.json()
        if not isinstance(rows, list):
            raise EarningsError(f"expected a list, got {type(rows).__name__}")
        if rows and not any(
            str(r.get("symbol", "")).upper() == symbol
            for r in rows if isinstance(r, dict)
        ):
            # Rows came back for other tickers: the endpoint is not honouring
            # `symbol`. Silently filtering to nothing here is the exact bug
            # this rewrite exists to remove.
            raise EarningsError(
                f"response contains no rows for {symbol} -- endpoint is not "
                f"symbol-scoped; check earnings.path"
            )

        upcoming = _earliest_upcoming(rows, symbol, today)
        return EarningsEntry(
            symbol=symbol,
            date=upcoming[0],
            confirmed=upcoming[1],
            fetched_at=fetched_at,
            source="fmp",
        )


def _confirmation(row: dict) -> bool | None:
    """Whether the provider says this date is scheduled, estimated, or neither.

    Returns ``None`` when the payload carries no confirmation signal at all --
    which is the case for the ``/stable/earnings`` shape, whose forward rows
    hold only ``epsEstimated``, ``revenueEstimated`` and ``lastUpdated``.
    Reporting ``False`` there would be a lie in the safe-looking direction: it
    reads as "the provider told us this is an estimate" when the provider told
    us nothing. The distinction matters because ``require_confirmed`` turns
    "not True" into an exclusion, and an operator flipping that flag deserves
    to know whether they are filtering estimates or filtering silence.
    """
    if row.get("epsActual") is not None or row.get("eps") is not None:
        return True                      # actuals exist -- the print is real
    raw_time = row.get("time")
    if raw_time is not None:
        return str(raw_time).lower() in ("bmo", "amc", "dmh")
    return None                          # no signal in this payload shape


def _earliest_upcoming(
    rows: Iterable[dict], symbol: str, today: date
) -> tuple[str | None, bool | None]:
    """Earliest dated row at or after today, plus whether it looks confirmed.

    FMP has shipped several shapes for this payload over the years, so the
    date field is read tolerantly. What is *not* tolerated is guessing: an
    unparseable row is skipped, and no rows means no date, which excludes.
    """
    best: date | None = None
    best_confirmed: bool | None = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("symbol", symbol)).upper() != symbol:
            continue
        raw = row.get("date") or row.get("earningsDate") or row.get("reportDate")
        if not raw:
            continue
        try:
            parsed = date.fromisoformat(str(raw)[:10])
        except ValueError:
            continue
        if parsed < today:
            continue
        confirmed = _confirmation(row)
        if best is None or parsed < best:
            best, best_confirmed = parsed, confirmed
    return (best.isoformat() if best else None, best_confirmed)


# ---------------------------------------------------------------------------
# The exclusion itself
# ---------------------------------------------------------------------------


def evaluate_exclusion(
    symbol: str,
    entry: EarningsEntry | None,
    order_session: date,
    trading_calendar: Any,
    now: datetime,
    max_hold_sessions: int,
    buffer_sessions: int,
    max_cache_age_hours: float,
    require_confirmed: bool = False,
    no_earnings: bool = False,
) -> EarningsVerdict:
    """Hold-window exclusion. Runs in code, before any model call, no override.

    Excludes any symbol whose next earnings date falls within
    ``max_hold_sessions + buffer_sessions`` trading sessions of the entry
    session. Every uncertain case excludes.

    ``no_earnings`` marks an instrument declared in
    ``config/universe.yaml: no_earnings`` as structurally print-free. Staleness
    is not checked for those: the claim comes from the config, not from the
    provider, so a stale fetch is not evidence against it. What *is* checked is
    the contradiction -- a declared-print-free symbol carrying a real date means
    the declaration is wrong, and a wrong declaration is more dangerous than a
    missing one because it reads as deliberate.
    """
    if no_earnings:
        if entry is not None and entry.date is not None:
            return EarningsVerdict(
                symbol, True,
                f"declared no-earnings but the provider returned {entry.date} -- "
                f"remove it from universe.yaml: no_earnings",
                earnings_date=entry.date, confirmed=entry.confirmed,
                no_earnings_class=True,
            )
        return EarningsVerdict(
            symbol, False, "no-earnings instrument, declared in universe.yaml",
            no_earnings_class=True,
        )

    if entry is None:
        return EarningsVerdict(symbol, True, "no earnings data for symbol")

    age = entry.age_hours(now)
    if entry.is_stale(now, max_cache_age_hours):
        return EarningsVerdict(
            symbol, True,
            f"earnings data is {age:.1f}h old, limit {max_cache_age_hours:.0f}h",
            earnings_date=entry.date, confirmed=entry.confirmed, age_hours=age,
        )

    if entry.date is None:
        return EarningsVerdict(
            symbol, True, "no earnings date known", confirmed=entry.confirmed, age_hours=age
        )

    if require_confirmed and entry.confirmed is not True:
        detail = (
            "the provider gave no confirmation signal"
            if entry.confirmed is None
            else "the provider marks it an estimate"
        )
        return EarningsVerdict(
            symbol, True, f"earnings date is not confirmed -- {detail}",
            earnings_date=entry.date, confirmed=entry.confirmed, age_hours=age,
        )

    horizon = max_hold_sessions + buffer_sessions
    sessions_until = trading_calendar.sessions_until(entry.as_date, order_session)

    if sessions_until < 0:
        return EarningsVerdict(
            symbol, True, "earnings date is in the past -- data is not current",
            earnings_date=entry.date, confirmed=entry.confirmed,
            sessions_until=sessions_until, age_hours=age,
        )

    if sessions_until <= horizon:
        return EarningsVerdict(
            symbol, True,
            f"earnings in {sessions_until} session(s), within the "
            f"{horizon}-session hold window",
            earnings_date=entry.date, confirmed=entry.confirmed,
            sessions_until=sessions_until, age_hours=age,
        )

    return EarningsVerdict(
        symbol, False, f"earnings {sessions_until} sessions out, clear of the window",
        earnings_date=entry.date, confirmed=entry.confirmed,
        sessions_until=sessions_until, age_hours=age,
    )


def assert_universe_resolves(
    symbols: Sequence[str],
    calendar: EarningsCalendar,
    no_earnings_symbols: Iterable[str],
    now: datetime,
    max_cache_age_hours: float,
) -> dict[str, str]:
    """Every symbol must resolve to a real date or an explicit no-earnings claim.

    Call this at startup, before the first session decision. The other
    fail-closed rules make a broken feed *safe* -- an unknown date excludes, so
    nothing trades on it. That safety is also what hides the break: a feed
    returning nothing for every symbol looks exactly like a quiet week. This
    assertion is the one place that refuses to accept silence as an answer.

    Returns a ``{symbol: resolution}`` map on success. Raises
    :class:`EarningsError` naming every symbol that resolved to neither, and
    every declared no-earnings symbol contradicted by a real date.
    """
    declared = {s.strip().upper() for s in no_earnings_symbols}
    resolved: dict[str, str] = {}
    unresolved: list[str] = []
    contradicted: list[str] = []

    for symbol in (s.strip().upper() for s in symbols):
        entry = calendar.get(symbol)
        if symbol in declared:
            if entry is not None and entry.date is not None:
                contradicted.append(f"{symbol} (declared no-earnings, got {entry.date})")
            else:
                resolved[symbol] = "no_earnings_class"
            continue
        if entry is None:
            unresolved.append(f"{symbol} (no entry -- never fetched)")
        elif entry.date is None:
            unresolved.append(f"{symbol} (fetched, no date returned)")
        elif entry.is_stale(now, max_cache_age_hours):
            unresolved.append(
                f"{symbol} (data {entry.age_hours(now):.1f}h old, "
                f"limit {max_cache_age_hours:.0f}h)"
            )
        else:
            resolved[symbol] = "earnings_date"

    problems = unresolved + contradicted
    if problems:
        raise EarningsError(
            f"{len(problems)} of {len(list(symbols))} universe symbol(s) did not "
            f"resolve to an earnings date or a declared no-earnings instrument: "
            + "; ".join(problems)
            + ". Refetch, or declare the instrument in universe.yaml: no_earnings. "
            "Do not widen the exclusion to make this pass."
        )
    return resolved


def spans_earnings(expiry: date, earnings: date | None) -> bool | None:
    """Contract-span test: does this expiry sit on or after the earnings date?

    Returns ``None`` when the earnings date is unknown -- the caller decides,
    and for the prefilter that means exclude.
    """
    if earnings is None:
        return None
    return expiry >= earnings
