"""Agent 2 against stubs. The earnings filter and truncation are the point."""
from __future__ import annotations

import time
from datetime import date, datetime, timezone

import pytest

from src.agents.a2_context import ContextInputs, ContextScreener
from src.agents.runner import AgentRunner

SESSION = date(2026, 8, 25)
NOW = datetime(2026, 8, 25, 13, 30, tzinfo=timezone.utc)

TEMPLATE = (
    "Assess $symbol at $spot. iv_vs_rv20 $iv_vs_rv20, iv_percentile $iv_percentile, "
    "atr% $atr_pct_of_spot, rv $realized_vol, trend $trend_pct_20d.\n"
    "Earnings: $sessions_until_earnings ahead, $sessions_since_earnings since.\n"
    "Headlines:\n$headlines\nObservations:\n$observations\n"
)


def good(symbol="AMD", eligible=True, bias=0.8, **kw):
    base = {
        "symbol": symbol, "eligible": eligible, "hard_blocks": [],
        "directional_bias": "bullish", "bias_strength": bias,
        "event_risk": "low", "iv_assessment": "fair", "notes": "",
    }
    base.update(kw)
    return base


@pytest.fixture
def config(tmp_path, monkeypatch):
    from src.config import load_config

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "a2_context.txt").write_text(TEMPLATE, encoding="utf-8")
    monkeypatch.setenv("DEEPSEES_PROMPT_DIR", str(prompts))
    return load_config()


class RecordingLog:
    def __init__(self):
        self.records = []

    def write(self, payload, action, **kw):
        self.records.append({"payload": payload, "action": action, **kw})

    def of_kind(self, kind):
        return [r for r in self.records if r["payload"].kind == kind]


@pytest.fixture
def screener(config):
    log = RecordingLog()
    runner = AgentRunner(config, decision_log=log)
    s = ContextScreener(config, runner)
    s.recording = log
    yield s
    runner.close()


def candidate(symbol="AMD", **kw):
    base = dict(symbol=symbol, spot=475.0, atr_pct_of_spot=0.03, realized_vol=0.4,
                iv_vs_rv20=1.1, iv_percentile=0.5, trend_pct_20d=0.05)
    base.update(kw)
    return ContextInputs(**base)


# --- the earnings filter runs in code, before the model --------------------


class FakeEarnings:
    """Only the .get() the screener uses."""

    def __init__(self, entries):
        self._entries = entries

    def get(self, symbol):
        return self._entries.get(symbol.upper())


def FakeCalendar():
    """A real TradingCalendar over synthetic weekday sessions.

    Not a stub: the post-print buffer counts sessions BACKWARDS to the previous
    print, so the window has to reach back past it. A fake that answers
    sessions_until() but carries no session list is exactly the shape of
    calendar CLAUDE.md warns produces a false exclusion.
    """
    from datetime import timedelta

    from src.brokers.alpaca.calendar import TradingCalendar

    start, end = date(2026, 4, 1), date(2026, 12, 31)
    days, cursor = [], start
    while cursor <= end:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return TradingCalendar(
        sessions=tuple(days),
        closes={d: datetime(d.year, d.month, d.day, 16, 0) for d in days},
    )


def _entry(next_date, prev_date="2026-05-20"):
    from src.earnings.calendar import EarningsEntry

    return EarningsEntry(
        symbol="X", date=next_date, previous_date=prev_date,
        fetched_at=NOW.isoformat(), confirmed=True,
    )


def test_an_excluded_symbol_is_never_shown_to_the_model(screener):
    """The whole design: the model cannot override a rule it never sees."""
    seen = []

    def capture(prompt, feedback):
        seen.append(prompt)
        return good("TESTA")

    # TESTA rather than SPY: SPY is declared no_earnings in the universe, and
    # handing a declared instrument a real date is a contradiction the
    # exclusion rejects loudly -- correctly, but it is not this test's subject.
    earnings = FakeEarnings({
        "AMD": _entry("2026-08-27"),        # 2 sessions out -- inside the window
        "TESTA": _entry("2026-12-01"),      # far away
    })
    res = screener.screen(
        [candidate("AMD"), candidate("TESTA")], SESSION, capture,
        earnings=earnings, trading_calendar=FakeCalendar(), now=NOW,
    )
    assert len(seen) == 1
    assert "AMD" not in seen[0]                      # never rendered at all
    assert [v.symbol for v in res.excluded_in_code] == ["AMD"]
    assert res.screened_symbols == ("TESTA",)


def test_the_exclusion_is_recorded_with_its_reason(screener):
    earnings = FakeEarnings({"AMD": _entry("2026-08-27")})
    res = screener.screen(
        [candidate("AMD")], SESSION, lambda p, f: good("AMD"),
        earnings=earnings, trading_calendar=FakeCalendar(), now=NOW,
    )
    [verdict] = res.excluded_in_code
    assert verdict.excluded is True
    assert verdict.reason
    assert res.eligible == ()


def test_a_model_calling_an_excluded_symbol_eligible_cannot_matter(screener):
    """Even a model insisting the symbol is fine never gets asked."""
    earnings = FakeEarnings({"AMD": _entry("2026-08-27")})
    res = screener.screen(
        [candidate("AMD")], SESSION,
        lambda p, f: good("AMD", eligible=True, bias=0.99),
        earnings=earnings, trading_calendar=FakeCalendar(), now=NOW,
    )
    assert res.symbols == ()


def test_without_a_calendar_the_model_path_still_runs(screener):
    """Optional only so the model path can be exercised on its own."""
    res = screener.screen([candidate("AMD")], SESSION, lambda p, f: good("AMD"))
    assert res.symbols == ("AMD",)
    assert res.excluded_in_code == ()


# --- truncation is not failure ---------------------------------------------


def test_a_surplus_eligible_set_is_truncated_not_failed(screener):
    """Twelve good answers against a cap of three is a truncation."""
    names = [f"S{i}" for i in range(6)]
    biases = {"S0": 0.9, "S1": 0.4, "S2": 0.8, "S3": 0.5, "S4": 0.95, "S5": 0.6}

    def by_symbol(prompt, feedback):
        sym = next(n for n in names if n in prompt)
        return good(sym, bias=biases[sym])

    res = screener.screen([candidate(n) for n in names], SESSION, by_symbol)
    assert res.symbols == ("S4", "S0", "S2")        # max_eligible_symbols = 3
    assert len(res.failed) == 0                     # nothing failed
    assert len(res.overrides) == 1


def test_truncation_logs_as_an_override_with_both_lists(screener):
    names = [f"S{i}" for i in range(5)]

    def by_symbol(prompt, feedback):
        sym = next(n for n in names if n in prompt)
        return good(sym, bias=0.9 - 0.1 * int(sym[1]))

    screener.screen([candidate(n) for n in names], SESSION, by_symbol)
    [rec] = screener.recording.of_kind("agent_override")
    payload = rec["payload"]
    assert payload.override == "force"              # no response was invalid
    assert payload.rule == "agents.a2.max_eligible_symbols"
    assert len(payload.model_value) == 5            # the full ranked list
    assert len(payload.applied_value) == 3          # what was kept
    assert rec["action"] == "agent_force"


def test_a_set_below_the_cap_is_not_truncated(screener):
    res = screener.screen([candidate("AMD")], SESSION, lambda p, f: good("AMD"))
    assert res.overrides == ()
    assert len(screener.recording.of_kind("agent_override")) == 0


def test_ineligible_symbols_do_not_consume_cap_slots(screener):
    names = ["A", "B", "C", "D"]

    def by_symbol(prompt, feedback):
        sym = next(n for n in names if f"Assess {n} " in prompt)
        return good(sym, eligible=sym in ("A", "B"), bias=0.9)

    res = screener.screen([candidate(n) for n in names], SESSION, by_symbol)
    assert set(res.symbols) == {"A", "B"}
    assert len(res.ineligible) == 2


# --- fail closed, per symbol ------------------------------------------------


def test_one_bad_response_costs_only_that_symbol(screener):
    """Unlike Agent 1, a failure here is not total."""
    def mixed(prompt, feedback):
        return "{not json" if "Assess AMD " in prompt else good("SPY")

    res = screener.screen([candidate("AMD"), candidate("SPY")], SESSION, mixed)
    assert res.symbols == ("SPY",)
    assert [s for s, _ in res.failed] == ["AMD"]


def test_malformed_output_blocks_that_symbol(screener):
    res = screener.screen([candidate("AMD")], SESSION, lambda p, f: "{not json")
    assert res.symbols == ()
    [(sym, run)] = res.failed
    assert sym == "AMD" and run.blocks_action is True


def test_an_empty_response_blocks_that_symbol(screener):
    res = screener.screen([candidate("AMD")], SESSION, lambda p, f: "")
    assert res.symbols == () and len(res.failed) == 1


def test_a_timeout_blocks_that_symbol(config):
    runner = AgentRunner(config)
    runner.timeout = 0.05
    s = ContextScreener(config, runner)

    def slow(prompt, feedback):
        time.sleep(0.5)
        return good("AMD")

    res = s.screen([candidate("AMD")], SESSION, slow)
    assert res.symbols == ()
    [(_, run)] = res.failed
    assert run.timed_out is True and run.blocks_action is True
    runner.close()


def test_out_of_range_bias_strength_fails_the_symbol(screener):
    res = screener.screen([candidate("AMD")], SESSION,
                          lambda p, f: good("AMD", bias=7.0))
    assert res.symbols == () and len(res.failed) == 1


# --- rules force ineligible rather than failing ----------------------------


def test_high_event_risk_forces_ineligible(screener):
    res = screener.screen([candidate("AMD")], SESSION,
                          lambda p, f: good("AMD", event_risk="high"))
    assert res.symbols == ()
    assert len(res.ineligible) == 1        # forced, not failed
    assert res.failed == ()


def test_a_named_hard_block_with_eligible_true_is_a_failure(screener):
    """Schema-level contradiction: resolved against the blocker."""
    res = screener.screen([candidate("AMD")], SESSION,
                          lambda p, f: good("AMD", hard_blocks=["halted"]))
    assert len(res.failed) == 1


def test_weak_bias_forces_ineligible(screener):
    res = screener.screen([candidate("AMD")], SESSION,
                          lambda p, f: good("AMD", bias=0.1))
    assert res.symbols == () and len(res.ineligible) == 1


# --- prompt handling --------------------------------------------------------


def test_earnings_cycle_context_reaches_the_prompt(screener):
    seen = {}

    def capture(prompt, feedback):
        seen["p"] = prompt
        return good("AMD")

    screener.screen([candidate("AMD", sessions_until_earnings=40,
                               sessions_since_earnings=15)], SESSION, capture)
    assert "40 ahead" in seen["p"] and "15 since" in seen["p"]


def test_unknown_earnings_context_renders_as_unknown(screener):
    seen = {}

    def capture(prompt, feedback):
        seen["p"] = prompt
        return good("AMD")

    screener.screen([candidate("AMD")], SESSION, capture)
    assert "unknown ahead" in seen["p"]


def test_the_prompt_never_reaches_the_log(screener):
    screener.screen([candidate("AMD")], SESSION, lambda p, f: good("AMD"))
    for rec in screener.recording.of_kind("agent_call"):
        assert "Assess AMD" not in rec["payload"].model_dump_json()
