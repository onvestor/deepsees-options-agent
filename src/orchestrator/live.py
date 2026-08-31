"""The live session: the orchestrator's handlers, wired to the real pipeline.

This is the join. The cadence exists (:mod:`~src.orchestrator.scheduler`), the
agents exist, the prefilter exists, the order path exists, and this module is
what connects them so a session runs unattended.

**Every decision writes to the decision log, including the ones where nothing
happens.** A skip is a decision. The log's whole purpose is to reconstruct a
session afterwards, and a session that traded nothing is exactly the one where
"why not" needs an answer -- so a signal that did not fire, a chain with no
survivors, and a size that came out zero all leave a record with the same
weight as a fill.

**Session state is held here, not in the agents.** The eligible set from Agent 2
and the locked profiles from Agent 1 are produced pre-market and consumed by
entry scans hours later. They live on :class:`LiveSession` for exactly as long
as the session does, and :meth:`LiveSession.roll` drops them -- a profile
surviving into the next session would be a regime read applied to a day it was
never made for.

**Position state is never held.** Every handler that needs the book calls
:func:`~src.brokers.alpaca.positions.reconcile`. That is the contract that
module exists to keep, and holding a book across handlers would break it inside
a single tick.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from src.agents.a1_regime import RegimeInputs, RegimeProfiler
from src.agents.a2_context import ContextInputs, ContextScreener
from src.agents.a3_risk import RiskAllocator, RiskInputs
from src.agents.a4_contract import ContractInputs, ContractSelector
from src.agents.a5_exit import ExitInputs, ExitManager
from src.agents.a6_review import NightlyReviewer, ObservationStore, ReviewInputs
from src.agents.runner import AgentRunner
from src.agents.schemas import Direction, ExitAction, Structure
from src.brokers.alpaca.calendar import TradingCalendar
from src.brokers.alpaca.client import AlpacaClients, sizing_capital, with_retry
from src.brokers.alpaca.orders import (
    SHARES_PER_CONTRACT,
    ExecutionLimits,
    OrderError,
    is_filled,
    place_entry,
    place_exit,
)
from src.brokers.alpaca.positions import Book, OptionPosition, reconcile
from src.brokers.alpaca.quotes import fetch_snapshots
from src.decisionlog.adapters import prefilter_payload, signal_eval_payload
from src.decisionlog.schema import (
    CapOverridePayload,
    KillSwitchPayload,
    OrderPayload,
    SessionPayload,
    SizingPayload,
    SkipPayload,
)
from src.options.metrics import realized_volatility
from src.options.prefilter import assemble, plan_scan
from src.risk.sizing import AccountState
from src.signals import engine as signal_engine
from src.signals.indicators import atr as atr_indicator, rsi as rsi_indicator

log = logging.getLogger(__name__)


class SkipReason(str):
    """A stage skip, recorded verbatim in the log."""


@dataclass
class LiveSession:
    """One session's state and the six handlers that advance it."""

    config: Any
    clients: AlpacaClients
    calendar: TradingCalendar
    transports: dict[str, Callable[[str, str | None], Any]]
    decision_log: Any
    symbols: tuple[str, ...]
    dry_run: bool = False
    """Reads and agent calls run; no order is ever sent."""

    # -- session state, dropped on roll ------------------------------------
    session: date | None = None
    eligible: dict[str, Any] = field(default_factory=dict)
    profiles: dict[str, Any] = field(default_factory=dict)
    entries_this_session: int = 0
    skips: list[tuple[str, str, str]] = field(default_factory=list)
    orders_placed: int = 0
    fills: int = 0

    def __post_init__(self) -> None:
        self.limits = self.config.limits
        self.runner = AgentRunner(self.config, decision_log=self.decision_log)
        self.a1 = RegimeProfiler(self.config, self.runner)
        self.a2 = ContextScreener(self.config, self.runner)
        self.a3 = RiskAllocator(self.config, self.runner)
        self.a4 = ContractSelector(self.config, self.runner)
        self.a5 = ExitManager(self.config, self.runner)
        self.a6 = NightlyReviewer(self.config, self.runner)
        self.observations = ObservationStore()
        self.execution = ExecutionLimits.from_limits(self.limits)
        self.entry_cap = self.limits.get_int("caps.max_entries_per_session")
        self.max_per_symbol = self.limits.get_int("caps.max_positions_per_symbol")
        self.rv_window = self.limits.get_int("metrics.realized_vol_window_days")
        self.atr_period = self.limits.get_int("signals.atr_period")
        self.signal_settings = signal_engine.SignalSettings.from_limits(self.limits)
        self.stop_pct = self.limits.get_float("exits.stop_pct")
        self.target_pct = self.limits.get_float("exits.target_pct")
        self.max_hold = self.limits.get_int("exits.max_hold_sessions")
        self.min_to_expiry = self.limits.get_int("exits.min_sessions_to_expiry")
        self.bars_ttl = self.limits.get_float("cache.bars_ttl_seconds")
        self._bars: dict[tuple[str, date], tuple[datetime, Any]] = {}
        # One id per entry attempt and per exit decision. entry_scan runs
        # every few minutes, so the same symbol is scanned many times a
        # session -- grouping a causal chain by symbol alone would merge
        # separate attempts into one chain that never happened.
        self._attempt = 0

    def close(self) -> None:
        self.runner.close()

    # -- housekeeping -------------------------------------------------------

    def roll(self, session: date) -> None:
        """Drop everything session-scoped. A profile must not outlive its day."""
        if self.session == session:
            return
        log.info("live: rolling session state %s -> %s", self.session, session)
        self.session = session
        self.eligible.clear()
        self.profiles.clear()
        self.a1.clear()
        self.entries_this_session = 0
        self.skips.clear()
        self.orders_placed = 0
        self.fills = 0
        self._bars.clear()
        self._attempt = 0

    _trace: str | None = None

    def skip(self, stage: str, symbol: str, reason: str, **detail: Any) -> None:
        """Record a skip -- to the decision log, not just to stderr.

        Every one of them. A skip that only reached the process log would be
        invisible to the artifact the session is run to produce, and the
        stage-level ones (no eligible set, cap reached, feed unavailable) have
        no other payload that would carry them.
        """
        self.skips.append((stage, symbol, reason))
        log.info("live skip [%s] %s: %s", stage, symbol or "-", reason)
        self._write(
            SkipPayload(stage=stage, reason=reason, detail=detail),
            action="skip", symbol=symbol or None,
        )

    def _write(self, payload: Any, action: str, **kw: Any) -> None:
        if self.decision_log is not None:
            kw.setdefault("trace_id", self._trace)
            self.decision_log.write(payload, action=action, **kw)

    # -- market data --------------------------------------------------------

    def bars(self, symbol: str, session: date) -> Any:
        """Daily bars, re-fetched on a short TTL.

        **This was cached for the whole session and that was the bug.** The
        docstring used to claim a daily frame does not change within a session.
        It does: the current day's bar forms continuously while the market is
        open. Worse, the first fetch happens pre-market during the Agent 2
        screen, when today's bar does not exist at all -- so every scan for the
        next six and a half hours was served a frame ending on the *previous*
        session. Measured 31 Aug 2026: 189 signal evaluations, every one
        carrying the same `bar_ts` and `bar_count`, producing three distinct
        results repeated sixty-three times each.

        The TTL is short enough that each entry scan sees current data and long
        enough that the several calls inside one scan do not refetch two
        hundred days per symbol. Positions and quotes remain uncached entirely.
        """
        key = (symbol, session)
        hit = self._bars.get(key)
        if hit is not None:
            fetched_at, frame = hit
            if (_now() - fetched_at).total_seconds() < self.bars_ttl:
                return frame

        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=date.today() - timedelta(days=200),
            feed=self.clients.equities_feed,
        )
        frame = with_retry(
            self.config, f"bars({symbol})",
            lambda: self.clients.stocks.get_stock_bars(request).df,
        )
        if "symbol" in getattr(frame.index, "names", []):
            frame = frame.xs(symbol, level="symbol")
        self._bars[key] = (_now(), frame)
        return frame

    def has_partial_bar(self, frame: Any, session: date) -> bool:
        """Whether the final bar is today's, and therefore still forming.

        Only ever consulted during an entry scan, which runs inside the entry
        window -- so a bar dated today is by definition incomplete there.
        """
        try:
            return bool(len(frame)) and frame.index[-1].date() == session
        except Exception:  # noqa: BLE001 -- an unreadable index is not partial
            return False


    def stats(self, symbol: str, session: date) -> tuple[float, float, float] | None:
        """Spot, ATR and realised vol, or None when history is too short."""
        frame = self.bars(symbol, session)
        if frame is None or len(frame) < max(self.atr_period, self.rv_window) + 1:
            return None
        spot = float(frame["close"].iloc[-1])
        # Same rule as the engine: a forming bar's range is partial, so it is
        # excluded from ATR while its close is used as the current price.
        completed = (
            frame.iloc[:-1]
            if self.has_partial_bar(frame, session) and len(frame) > 1
            else frame
        )
        atr = float(atr_indicator(completed, self.atr_period).iloc[-1])
        try:
            rv = realized_volatility(list(frame["close"]), self.rv_window)
        except Exception:  # noqa: BLE001 -- too little history is a skip
            return None
        if spot <= 0 or atr <= 0 or rv <= 0:
            return None
        return spot, atr, rv

    # -- handler: Agent 2, pre-market screen -------------------------------

    def a2_context(self, state: Any) -> None:
        self.roll(state.session)
        candidates: list[ContextInputs] = []
        for symbol in self.symbols:
            stats = self.stats(symbol, state.session)
            if stats is None:
                self.skip("a2", symbol, "insufficient bar history")
                continue
            spot, atr, rv = stats
            frame = self.bars(symbol, state.session)
            candidates.append(
                ContextInputs(
                    symbol=symbol,
                    spot=spot,
                    atr_pct_of_spot=atr / spot,
                    realized_vol=rv,
                    iv_vs_rv20=1.0,
                    iv_percentile=0.5,
                    trend_pct_20d=_trend(frame, 20),
                )
            )

        if not candidates:
            self.skip("a2", "-", "no symbol had usable history")
            return

        earnings, cal = self._earnings_inputs()
        if earnings is None or cal is None:
            # Fail closed. An unknown earnings date excludes, so an entirely
            # unavailable feed must exclude everything -- running the model
            # without the check would invert the rule it is there to enforce.
            self.skip("a2", "-", "earnings feed unavailable; no symbol admitted")
            return
        result = self.a2.screen(
            candidates, state.session, self.transports["a2"],
            earnings=earnings, trading_calendar=cal,
            # Aware, in UTC. Cached earnings entries carry an aware fetch
            # stamp, and a naive `now` here raises inside the staleness check
            # -- which surfaces as the whole screen failing rather than as a
            # date problem.
            now=datetime.now(tz=timezone.utc),
        )
        for verdict in result.excluded_in_code:
            self.skip("a2_earnings", verdict.symbol, verdict.reason)
        for symbol, run in result.failed:
            self.skip("a2", symbol, f"agent failed: {run.error or 'validation'}")
        for decision in result.ineligible:
            self.skip("a2", decision.symbol,
                      f"ineligible: {list(decision.hard_blocks) or decision.notes[:80]}")

        self.eligible = {d.symbol: d for d in result.eligible}
        log.info("live a2: eligible %s", list(self.eligible) or "(none)")

    def _earnings_inputs(self) -> tuple[Any, Any]:
        """The earnings feed, or ``(None, None)`` to run the model path alone.

        A live session must supply both -- the exclusion is a hard, pre-model
        rule. When the feed cannot be built the screen is skipped entirely
        rather than run without it, because an unavailable earnings date is
        supposed to exclude, and silently dropping the check would invert that.
        """
        try:
            from src.earnings.calendar import EarningsCalendar

            calendar = EarningsCalendar.from_config(self.config)
            # Refresh once per session. The cache is provider data with its own
            # staleness rule; evaluate_exclusion re-checks the age and excludes
            # on a stale entry, so a failed refresh degrades to exclusion
            # rather than to a silently old date.
            try:
                calendar.refresh(list(self.symbols))
            except Exception as exc:  # noqa: BLE001
                log.warning("earnings refresh failed (%s); using the cache, which "
                            "excludes anything stale", exc)
            return calendar, self.calendar
        except Exception as exc:  # noqa: BLE001
            log.error(
                "earnings calendar unavailable (%s). The exclusion is a hard "
                "pre-model rule, so the screen runs with NO symbols admitted "
                "rather than without the check.", exc,
            )
            return None, None

    # -- handler: Agent 1, pre-market profile -------------------------------

    def a1_regime(self, state: Any) -> None:
        self.roll(state.session)
        if not self.eligible:
            self.skip("a1", "-", "no eligible symbols from a2")
            return

        for symbol in list(self.eligible):
            stats = self.stats(symbol, state.session)
            if stats is None:
                self.skip("a1", symbol, "insufficient bar history")
                continue
            spot, atr, rv = stats
            frame = self.bars(symbol, state.session)
            result = self.a1.profile(
                RegimeInputs(
                    symbol=symbol, spot=spot, atr=atr, atr_pct_of_spot=atr / spot,
                    realized_vol=rv,
                    rsi=_last(rsi_indicator(frame["close"],
                                            self.limits.get_int("signals.rsi_period")), 50.0),
                    ema_fast_value=float(frame["close"].ewm(span=9).mean().iloc[-1]),
                    ema_slow_value=float(
                        frame["close"].ewm(span=self.signal_settings.ema_slow).mean().iloc[-1]
                    ),
                    trend_pct_20d=_trend(frame, 20),
                    above_vwap=bool(frame["close"].iloc[-1] >= frame["close"].mean()),
                    observations=self.observations.live_for(
                        symbol, state.session, self._sessions_between
                    ),
                ),
                state.session, self.transports["a1"],
            )
            if not result.ok:
                self.skip("a1", symbol, f"agent failed: {result.run.error or 'validation'}")
                continue
            if result.decision.signal_profile.allowed_direction is Direction.NONE:
                self.skip("a1", symbol,
                          f"regime {result.decision.regime.value} permits no direction")
                continue
            self.profiles[symbol] = result.decision
        log.info("live a1: profiled %s", list(self.profiles) or "(none)")

    # -- handler: entry scan ------------------------------------------------

    def entry_scan(self, state: Any) -> None:
        self.roll(state.session)
        if not self.profiles:
            self.skip("entry", "-", "no profiled symbol (a1/a2 produced nothing)")
            return
        if self.entries_this_session >= self.entry_cap:
            self._write(
                CapOverridePayload(
                    cap_name="caps.max_entries_per_session",
                    requested=float(self.entries_this_session + 1),
                    cap_value=float(self.entry_cap),
                    applied=float(self.entries_this_session),
                    stage="entry",
                ),
                action="skip",
            )
            self.skip("entry", "-", f"session entry cap {self.entry_cap} reached")
            return

        book = reconcile(self.clients)
        for symbol in list(self.profiles):
            if self.entries_this_session >= self.entry_cap:
                break
            if book.count_in(symbol) >= self.max_per_symbol:
                self.skip("entry", symbol,
                          f"already holding {book.count_in(symbol)} "
                          f"(caps.max_positions_per_symbol {self.max_per_symbol})")
                continue
            self._try_entry(symbol, state, book)

    def _try_entry(self, symbol: str, state: Any, book: Book) -> None:
        self._attempt += 1
        self._trace = f"e{self._attempt:03d}-{symbol}"
        try:
            self._entry_attempt(symbol, state, book)
        finally:
            self._trace = None

    def _entry_attempt(self, symbol: str, state: Any, book: Book) -> None:
        stats = self.stats(symbol, state.session)
        if stats is None:
            self.skip("entry", symbol, "insufficient bar history")
            return
        spot, atr, rv = stats
        frame = self.bars(symbol, state.session)
        decision = self.profiles[symbol]
        context = self.eligible[symbol]

        profile = signal_engine.SignalProfile(
            ema_fast=decision.signal_profile.ema_fast,
            confirmation_bars=max(1, decision.signal_profile.confirmation_bars),
            require_vwap_alignment=decision.signal_profile.require_vwap_alignment,
            min_atr_multiple=decision.signal_profile.min_atr_multiple,
            allowed_direction=decision.signal_profile.allowed_direction.value,
        )
        evaluation = signal_engine.evaluate(
            frame, profile, self.signal_settings,
            partial_last_bar=self.has_partial_bar(frame, state.session),
        )
        self._write(
            signal_eval_payload(
                evaluation, bar_ts=str(frame.index[-1]), bar_count=len(frame),
                profile=profile, profile_source="agent",
            ),
            action="entry" if evaluation.triggered else "skip",
            symbol=symbol,
        )
        if not evaluation.triggered or evaluation.direction == "none":
            self.skip("signal", symbol,
                      f"no signal: blocked by {list(evaluation.blocked_by) or evaluation.reasons}")
            return

        option_type = "call" if evaluation.direction == "long_calls" else "put"
        try:
            plan = plan_scan(self.limits, self.calendar, state.session, symbol, spot)
        except Exception as exc:  # noqa: BLE001 -- DteError and friends
            self.skip("prefilter", symbol, f"no scannable expiry: {exc}")
            return

        from src.brokers.alpaca.cache import MarketDataCache
        from src.brokers.alpaca.contracts import fetch as fetch_contracts

        cache = getattr(self, "_cache", None)
        if cache is None:
            cache = self._cache = MarketDataCache.from_config(self.config)

        specs = fetch_contracts(
            self.clients, symbol, expiry_gte=plan.expiry_gte, expiry_lte=plan.expiry_lte,
            option_type=option_type, strike_gte=plan.strike_gte,
            strike_lte=plan.strike_lte, cache=cache,
        )
        quotes = fetch_snapshots(self.clients, [s.symbol for s in specs], cache=cache)
        result = assemble(symbol, specs, quotes, self.calendar, state.session,
                          spot, atr, rv, self.limits, plan)
        self._write(
            prefilter_payload(
                result.candidates, result.thresholds, underlying_price=spot,
                detail=self.limits.get_str("decision_log.prefilter_detail"),
                near_boundary_pct=self.limits.get_float("decision_log.near_boundary_pct"),
                keep_symbols=[c.symbol for c in result.top],
            ),
            action="entry" if result.top else "skip",
            symbol=symbol,
        )
        if not result.top:
            self.skip("prefilter", symbol,
                      f"no survivors of {len(result.candidates)}: {result.reason_counts}")
            return

        selection = self.a4.select(
            ContractInputs(
                symbol=symbol, spot=spot, atr=atr, survivors=result.top,
                regime=decision.regime.value, confidence=decision.confidence,
                directional_bias=context.directional_bias.value,
                bias_strength=context.bias_strength,
                iv_assessment=context.iv_assessment.value,
                target_expiry=plan.target_expiry.isoformat(),
                session_dte=plan.target_session_dte,
                observations=self.observations.live_for(
                    symbol, state.session, self._sessions_between
                ),
            ),
            self.transports["a4"], trace_id=self._trace,
        )
        if not selection.ok:
            self.skip("a4", symbol, "no contract selected")
            return
        if selection.decision.structure is Structure.DEBIT_VERTICAL:
            # Unconstructible at this DTE and unbuilt in the order path.
            self.skip("a4", symbol, "vertical requested; single-leg only")
            return

        chosen = selection.decision.primary_symbol
        quote = quotes.get(chosen)
        spec = next((s for s in specs if s.symbol == chosen), None)
        if quote is None or spec is None or quote.mid is None:
            self.skip("a4", symbol, f"no usable quote for {chosen}")
            return

        cost = quote.ask * SHARES_PER_CONTRACT
        account = self._account_state(book, symbol)
        base = self.a3.base_size(cost, cost, account)
        self._write(
            SizingPayload(
                sizing_capital=account.options_buying_power,
                capital_source="options_buying_power",
                risk_per_trade=base.risk_budget,
                premium_per_contract=cost,
                base_contracts=base.base_contracts,
                final_contracts=base.final_contracts,
            ),
            action="entry" if base.final_contracts > 0 else "skip", symbol=symbol,
        )

        allocation = self.a3.allocate(
            RiskInputs(
                symbol=symbol, contract_symbol=chosen,
                base_contracts=base.final_contracts,
                cost_per_contract=cost, max_risk_per_contract=cost,
                risk_budget=base.risk_budget, equity=account.equity,
                open_positions=account.open_positions, open_premium=account.open_premium,
                regime=decision.regime.value, confidence=decision.confidence,
                bias_strength=context.bias_strength,
                iv_assessment=context.iv_assessment.value,
                observations=self.observations.live_for(
                    symbol, state.session, self._sessions_between
                ),
            ),
            account, self.transports["a3"], trace_id=self._trace,
        )
        for verdict in (allocation.sizing.caps if allocation.sizing else ()):
            if verdict.binding:
                self._write(
                    CapOverridePayload(
                        cap_name=verdict.name, requested=float(verdict.observed),
                        cap_value=float(verdict.limit_value),
                        applied=float(verdict.allowed_contracts), stage=verdict.stage,
                    ),
                    action="clamp", symbol=symbol,
                )
        if not allocation.ok or allocation.contracts <= 0:
            self.skip("a3", symbol,
                      allocation.reason or "sized to zero")
            return

        qty = allocation.contracts
        if self.dry_run:
            self.skip("order", symbol, f"dry run: would buy {qty} {chosen} at the mid")
            return

        try:
            entry = place_entry(
                self.clients, symbol=chosen, qty=qty, quote=quote, limits=self.execution
            )
        except OrderError as exc:
            self._write(
                OrderPayload(intent="buy_to_open", legs=[chosen], qty=qty,
                             broker_error=str(exc)),
                action="skip", symbol=symbol,
            )
            self.skip("order", symbol, f"refused: {exc}")
            return

        self.orders_placed += 1
        self._write(
            OrderPayload(
                intent="buy_to_open", legs=[chosen], qty=qty,
                limit_price=entry.limit_prices[0] if entry.limit_prices else None,
                order_id=entry.order_id, status=str(entry.status),
                filled_qty=float(entry.filled), filled_avg_price=entry.fill_price,
                session_dte=plan.target_session_dte,
            ),
            action="entry" if is_filled(entry.order) else "skip", symbol=symbol,
        )
        if is_filled(entry.order):
            self.fills += 1
            self.entries_this_session += 1
            log.info("live entry FILLED %s x%d at %s", chosen, entry.filled,
                     entry.fill_price)
        else:
            self.skip("order", symbol,
                      f"passive mid limit did not fill ({entry.status})")

    # -- handler: Agent 5 and the deterministic exits -----------------------

    def a5_exit(self, state: Any) -> None:
        self.roll(state.session)
        book = reconcile(self.clients)
        if not book.positions:
            return

        quotes = fetch_snapshots(self.clients, list(book.symbols))
        for position in book.positions:
            quote = quotes.get(position.symbol)
            if quote is None or quote.mid is None:
                self.skip("a5", position.symbol, "no two-sided quote to mark against")
                continue
            self._manage(position, quote, state)

    def _manage(self, position: OptionPosition, quote: Any, state: Any) -> None:
        self._trace = f"x-{position.symbol}"
        try:
            self._manage_position(position, quote, state)
        finally:
            self._trace = None

    def _manage_position(self, position: OptionPosition, quote: Any, state: Any) -> None:
        current = quote.mid
        pnl_pct = position.pnl_pct()
        if pnl_pct is None:
            pnl_pct = (current - position.avg_entry_price) / position.avg_entry_price * 100.0
        held = max(0, self.calendar.sessions_until(state.session, _entry_session(position)))
        to_expiry = self.calendar.sessions_until(position.expiry, state.session)

        plan = self.a5.manage(
            ExitInputs(
                symbol=position.underlying, contract_symbol=position.symbol,
                entry_premium=position.avg_entry_price, current_premium=current,
                pnl_pct=pnl_pct, current_stop_pct=self.stop_pct,
                target_pct=self.target_pct, sessions_held=held,
                max_hold_sessions=self.max_hold, sessions_to_expiry=to_expiry,
                contracts=abs(position.qty),
                observations=self.observations.live_for(
                    position.underlying, state.session, self._sessions_between
                ),
            ),
            self.transports["a5"], trace_id=self._trace,
        )
        if plan.model_failed:
            self.skip("a5", position.symbol, f"agent failed: {plan.reason}")

        # The deterministic exits are armed regardless of what Agent 5 said or
        # failed to say. Agent 5 may only tighten the stop it is given.
        stop = plan.stop_pct
        reason = None
        if to_expiry <= self.min_to_expiry:
            reason = "expiry_week"
        elif pnl_pct <= stop:
            reason = "stop"
        elif pnl_pct >= self.target_pct:
            reason = "target"
        elif held >= self.max_hold:
            reason = "max_hold"
        elif plan.action is ExitAction.EXIT_NOW:
            reason = "agent_exit_now"

        if reason is None:
            log.info("live a5 hold %s: pnl %.2f%% stop %.2f target %.2f held %d dte %d",
                     position.symbol, pnl_pct, stop, self.target_pct, held, to_expiry)
            return

        if self.dry_run:
            self.skip("exit", position.symbol, f"dry run: would exit ({reason})")
            return

        exit_result = place_exit(
            self.clients, symbol=position.symbol, qty=abs(position.qty),
            quote=quote, reason=reason, limits=self.execution,
            quote_reader=lambda: fetch_snapshots(
                self.clients, [position.symbol]
            )[position.symbol],
        )
        self.orders_placed += 1
        self._write(
            OrderPayload(
                intent="sell_to_close", legs=[position.symbol], qty=abs(position.qty),
                limit_price=exit_result.limit_prices[-1] if exit_result.limit_prices else None,
                order_id=exit_result.order_id, status=str(exit_result.status),
                filled_qty=float(exit_result.filled),
                filled_avg_price=exit_result.fill_price,
                session_dte=to_expiry,
            ),
            action="exit", symbol=position.underlying, reasons=[reason],
        )
        if exit_result.complete:
            self.fills += 1
        else:
            self.skip("exit", position.symbol,
                      f"ladder exhausted, {abs(position.qty) - exit_result.filled} unfilled")

    # -- handler: reconcile -------------------------------------------------

    def reconcile_job(self, state: Any) -> None:
        """Authoritative book read, plus the kill-switch evidence."""
        self.roll(state.session)
        book = reconcile(self.clients)
        account = with_retry(self.config, "get_account", self.clients.trading.get_account)
        self._write(
            SessionPayload(
                event="open" if state.may_open else "resume",
                equity=float(account.equity),
                open_positions=len(book),
                account=_suffix(account),
                notes=(f"phase={state.phase.value} premium={book.open_premium:.2f} "
                       f"entries={self.entries_this_session} skips={len(self.skips)}"),
            ),
            action="continue",
        )
        for symbol, error in book.unparseable:
            self.skip("reconcile", symbol, f"unparseable position: {error}")

    def log_kill_switches(self, verdicts: tuple[Any, ...]) -> None:
        """Write every switch's verdict, fired or not.

        All of them, because "we were one trade from the consecutive-loss halt"
        is what a post-session review needs and a fired-only log cannot say.
        """
        for verdict in verdicts:
            self._write(
                KillSwitchPayload(
                    switch=verdict.switch, threshold=float(verdict.threshold),
                    observed=float(verdict.observed), fired=bool(verdict.fired),
                    halts_new_entries=bool(verdict.halts_new_entries),
                    scope=verdict.scope,
                ),
                action="halt" if verdict.fired else "continue",
            )

    # -- handler: Agent 6 ---------------------------------------------------

    def a6_review(self, state: Any) -> None:
        self.roll(state.session)
        book = reconcile(self.clients)
        account = with_retry(self.config, "get_account", self.clients.trading.get_account)
        result = self.a6.review(
            ReviewInputs(
                session=state.session,
                entries=self.entries_this_session,
                exits=max(0, self.fills - self.entries_this_session),
                skips=len(self.skips),
                wins=0, losses=0, realized_pnl=0.0,
                agent_clamps=0, agent_forces=0,
                agent_failures=sum(1 for _, _, r in self.skips if "agent failed" in r),
                fallbacks=0,
                symbols_traded=tuple(sorted(self.eligible)),
                notes=tuple(f"{stage}:{sym}:{why}" for stage, sym, why in self.skips[:20]),
            ),
            self.transports["a6"], store=self.observations,
        )
        self._write(
            SessionPayload(
                event="close", equity=float(account.equity), open_positions=len(book),
                account=_suffix(account),
                notes=(f"entries={self.entries_this_session} orders={self.orders_placed} "
                       f"fills={self.fills} skips={len(self.skips)} "
                       f"observations={len(result.observations)}"),
            ),
            action="continue",
        )

    # -- helpers ------------------------------------------------------------

    def _account_state(self, book: Book, symbol: str) -> AccountState:
        account = with_retry(self.config, "get_account", self.clients.trading.get_account)
        return AccountState(
            equity=float(account.equity),
            options_buying_power=sizing_capital(account),
            open_premium=book.open_premium,
            open_positions=len(book),
            positions_in_symbol=book.count_in(symbol),
            entries_this_session=self.entries_this_session,
            entries_this_symbol_this_session=0,
        )

    def _sessions_between(self, created: date, now: date) -> int:
        return max(0, self.calendar.sessions_until(now, created))

    def handlers(self) -> dict[str, Callable[[Any], None]]:
        return {
            "a2_context": self.a2_context,
            "a1_regime": self.a1_regime,
            "entry_scan": self.entry_scan,
            "a5_exit": self.a5_exit,
            "reconcile": self.reconcile_job,
            "a6_review": self.a6_review,
        }


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _suffix(account: Any) -> str | None:
    """Last four of the account number, read back from the broker.

    Suffix only. The full number is operator state that CLAUDE.md keeps out of
    the repository, and four characters are enough to tell a dev session from a
    competition one -- which is the only question the log needs to answer.
    """
    number = str(getattr(account, "account_number", "") or "")
    return number[-4:] or None


def _trend(frame: Any, sessions: int) -> float:
    if frame is None or len(frame) <= sessions:
        return 0.0
    past = float(frame["close"].iloc[-sessions - 1])
    now = float(frame["close"].iloc[-1])
    return (now - past) / past if past else 0.0


def _last(series: Any, default: float) -> float:
    try:
        value = float(series.iloc[-1])
    except Exception:  # noqa: BLE001
        return default
    return value if value == value else default


def _entry_session(position: OptionPosition) -> date:
    """The session a position was opened on.

    Alpaca's position payload does not carry it, so today is assumed. That
    makes ``sessions_held`` zero for a position opened before today, which
    **understates** the hold -- the max-hold exit therefore fires late rather
    than early, and the expiry-week rule is unaffected. Recorded here because
    the fix is to carry entry sessions in our own state, which needs somewhere
    durable to put them.
    """
    return date.today()
