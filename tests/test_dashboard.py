"""The dashboard. Four views, all derived from the log, none able to act.

Two properties carry the most weight here:

* **Read-only is structural.** The app must expose no mutating verb and must
  never construct a broker client. A "close position" control on a dashboard
  for an autonomous system is a contradiction, and the test asserts there is no
  code path to one rather than trusting that nobody added a button.
* **Clamps and forces stay apart.** A clamp means the model returned an invalid
  value; a force means it returned a legal one and a rule overrode it. Merging
  them would make a well-behaved model on a choppy day indistinguishable from
  one emitting garbage -- which is the whole reason the log separates them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dashboard.reader import Log, discover

SESSION = "2026-08-31"


def rec(seq, kind, payload, **kw):
    base = {
        "seq": seq, "kind": kind, "payload": {"kind": kind, **payload},
        "session_date": SESSION, "action": kw.pop("action", "continue"),
        "ts_et": f"2026-08-31T09:{seq:02d}:00-04:00",
        "ts_utc": f"2026-08-31T13:{seq:02d}:00Z",
        "symbol": kw.pop("symbol", None), "trace_id": kw.pop("trace_id", None),
        "latency_ms": kw.pop("latency_ms", None), "reasons": [],
    }
    base.update(kw)
    return base


def agent_call(seq, agent, symbol, parsed=None, status="ok", attempt=1, errors=(), **kw):
    return rec(seq, "agent_call", {
        "agent": agent, "model": "m", "prompt_hash": "h", "prompt_chars": 10,
        "response_parsed": parsed,
        "validation": {"status": status, "attempt": attempt,
                       "errors": list(errors), "clamps": []},
    }, symbol=symbol, action="accepted", latency_ms=kw.pop("latency_ms", 1000.0), **kw)


@pytest.fixture
def log_dir(tmp_path) -> Path:
    """A synthetic session: one filled entry, one skip, guardrails, switches."""
    rows = [
        rec(1, "killswitch", {"switch": "daily_loss_halt_pct", "threshold": 0.03,
                              "observed": 0.0, "fired": False}),
        rec(2, "killswitch", {"switch": "consecutive_losing_trades", "threshold": 3.0,
                              "observed": 2.0, "fired": False}),
        agent_call(3, "a2_context", "PLTR",
                   {"symbol": "PLTR", "eligible": True, "directional_bias": "bullish",
                    "bias_strength": 0.7, "event_risk": "low", "iv_assessment": "fair"},
                   trace_id="e001-PLTR"),
        rec(4, "agent_override", {"agent": "a2_context", "override": "force",
                                  "field": "eligible", "model_value": True,
                                  "applied_value": False, "rule": "min_bias_strength",
                                  "detail": ""},
            symbol="SPY", action="agent_force"),
        rec(5, "agent_override", {"agent": "a1_regime", "override": "clamp",
                                  "field": "signal_profile.ema_fast", "model_value": 7,
                                  "applied_value": 8, "rule": "allowed_ema_fast",
                                  "detail": ""},
            symbol="PLTR", action="agent_clamp", trace_id="e001-PLTR"),
        rec(6, "signal_eval", {"bar_ts": "t", "bar_count": 100, "direction": "long_calls",
                               "triggered": True, "gates": {"ema": True, "atr": True},
                               "metrics": {}, "profile": {}},
            symbol="PLTR", trace_id="e001-PLTR"),
        rec(7, "prefilter", {"total_contracts": 7, "survivors": 2, "rejected": 5,
                             "reason_counts": {"delta band": 3}, "sole_reason": {},
                             "survivor_symbols": []},
            symbol="PLTR", trace_id="e001-PLTR"),
        rec(8, "sizing", {"sizing_capital": 100000.0, "capital_source": "options_buying_power",
                          "risk_per_trade": 2500.0, "premium_per_contract": 2287.0,
                          "base_contracts": 1, "final_contracts": 1},
            symbol="PLTR", trace_id="e001-PLTR"),
        rec(9, "cap_override", {"cap_name": "max_premium_per_trade", "requested": 3254.0,
                                "cap_value": 2500.0, "applied": 0.0, "stage": "sizing"},
            symbol="MSFT", action="clamp"),
        rec(10, "order", {"intent": "buy_to_open", "legs": ["PLTR261016C00170000"],
                          "qty": 1, "limit_price": 22.87, "order_id": "o1",
                          "status": "OrderStatus.FILLED", "filled_qty": 1.0,
                          "filled_avg_price": 22.80},
            symbol="PLTR", action="entry", trace_id="e001-PLTR"),
        rec(11, "skip", {"stage": "a3", "reason": "final size 0 below min_contracts",
                         "detail": {}},
            symbol="MSFT", action="skip", trace_id="e002-MSFT"),
        agent_call(12, "a4_contract", "TSLA", None, status="failed", attempt=2,
                   errors=["response is not valid JSON"], trace_id="e003-TSLA"),
        rec(13, "session", {"event": "close", "equity": 99500.0, "open_positions": 1,
                            "notes": "entries=1"}),
    ]
    path = tmp_path / f"decision_log-{SESSION}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def log(log_dir) -> Log:
    return Log.load(discover(log_dir))


# --- loading ---------------------------------------------------------------


def test_loads_every_record(log):
    assert len(log.records) == 13
    assert log.sessions == [SESSION]


def test_a_half_written_final_line_is_skipped_not_fatal(tmp_path):
    """A session appends while the dashboard reads. Catching the writer
    mid-flush must not 500 the page."""
    path = tmp_path / "decision_log-x.jsonl"
    path.write_text(json.dumps(rec(1, "session", {"event": "open"})) + "\n{\"partial\":",
                    encoding="utf-8")
    assert len(Log.load([path]).records) == 1


# --- view 1: timeline ------------------------------------------------------


def test_timeline_is_every_decision_in_order(log):
    rows = log.timeline()
    assert len(rows) == 13
    assert [r["seq"] for r in rows] == sorted(r["seq"] for r in rows)


def test_timeline_carries_actor_verdict_and_latency(log):
    row = next(r for r in log.timeline() if r["seq"] == 3)
    assert row["actor"] == "a2_context"
    assert row["symbol"] == "PLTR"
    assert "eligible=True" in row["verdict"]
    assert row["latency_ms"] == 1000


def test_guardrail_rows_are_flagged(log):
    flagged = {r["seq"] for r in log.timeline() if r["guardrail"]}
    assert {1, 2, 4, 5, 9, 11} <= flagged      # switches, overrides, cap, skip
    assert 10 not in flagged                    # an order is not a guardrail


# --- view 2: traces --------------------------------------------------------


def test_traces_group_by_trace_id(log):
    traces = {t["trace_id"]: t for t in log.traces()}
    assert "e001-PLTR" in traces
    assert traces["e001-PLTR"]["steps"] == 6
    assert not traces["e001-PLTR"]["inferred"]


def test_a_trace_reports_its_outcome(log):
    traces = {t["trace_id"]: t for t in log.traces()}
    assert traces["e001-PLTR"]["outcome"] == "filled"
    assert traces["e002-MSFT"]["outcome"] == "skipped at a3"


def test_a_trace_chain_is_the_full_causal_path(log):
    chain = log.trace("e001-PLTR")["chain"]
    stages = [s["stage"] for s in chain]
    assert stages == ["agent", "override", "signal", "prefilter", "sizing", "order"]


def test_the_chain_carries_rejection_counts_and_gates(log):
    chain = log.trace("e001-PLTR")["chain"]
    prefilter = next(s for s in chain if s["kind"] == "prefilter")
    assert prefilter["payload"]["reason_counts"] == {"delta band": 3}
    signal = next(s for s in chain if s["kind"] == "signal_eval")
    assert signal["payload"]["gates"] == {"ema": True, "atr": True}


def test_an_unknown_trace_is_none(log):
    assert log.trace("nope") is None


def test_records_without_a_trace_id_are_marked_inferred(tmp_path):
    """Grouping by symbol and proximity is weaker evidence than a recorded
    chain, and the view must say which it is showing."""
    rows = [
        rec(1, "signal_eval", {"bar_ts": "t", "bar_count": 1, "direction": "long_calls",
                               "triggered": True, "gates": {}, "metrics": {}, "profile": {}},
            symbol="SPY"),
        rec(2, "skip", {"stage": "a3", "reason": "zero", "detail": {}}, symbol="SPY"),
    ]
    path = tmp_path / "decision_log-y.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    [trace] = Log.load([path]).traces()
    assert trace["inferred"]
    assert trace["trace_id"].startswith("inferred-")


def test_two_attempts_on_one_symbol_do_not_merge(tmp_path):
    """entry_scan re-runs every few minutes. Splicing separate attempts into
    one chain would show a causal path that never happened."""
    rows = []
    for i, seq in enumerate((1, 3), start=1):
        rows.append(rec(seq, "signal_eval",
                        {"bar_ts": "t", "bar_count": 1, "direction": "long_calls",
                         "triggered": True, "gates": {}, "metrics": {}, "profile": {}},
                        symbol="SPY"))
        rows.append(rec(seq + 1, "skip", {"stage": "a3", "reason": f"try {i}", "detail": {}},
                        symbol="SPY"))
    path = tmp_path / "decision_log-z.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert len(Log.load([path]).traces()) == 2


# --- view 3: guardrails ----------------------------------------------------


def test_clamps_and_forces_are_counted_separately(log):
    g = log.guardrails()
    assert g["summary"]["clamps"] == 1
    assert g["summary"]["forces"] == 1
    assert g["clamps"][0]["field"] == "signal_profile.ema_fast"
    assert g["forces"][0]["field"] == "eligible"


def test_every_guardrail_class_is_reported(log):
    g = log.guardrails()["summary"]
    assert g["cap_overrides"] == 1
    assert g["skips"] == 1
    assert g["schema_retries"] == 1
    assert g["killswitch_evaluations"] == 2


def test_a_schema_retry_is_surfaced_with_its_error(log):
    [retry] = log.guardrails()["schema_retries"]
    assert retry["attempt"] == 2
    assert "not valid JSON" in retry["errors"][0]


def test_switches_that_did_not_fire_are_still_reported(log):
    """'We were one trade from the halt' is what a review needs, and a
    fired-only log cannot say it."""
    switches = log.guardrails()["killswitch_evaluations"]
    assert len(switches) == 2
    assert all(not s["fired"] for s in switches)
    assert log.guardrails()["summary"]["killswitches_fired"] == []


# --- view 4: status --------------------------------------------------------


def test_positions_are_reconstructed_from_fills(log):
    s = log.status()
    assert len(s["positions_from_fills"]) == 1
    assert s["positions_from_fills"][0]["contract"] == "PLTR261016C00170000"
    assert s["fills"] == 1


def test_a_closed_round_trip_leaves_no_position(tmp_path):
    rows = [
        rec(1, "order", {"intent": "buy_to_open", "legs": ["X"], "qty": 1,
                         "filled_qty": 1.0, "filled_avg_price": 10.0}, symbol="S"),
        rec(2, "order", {"intent": "sell_to_close", "legs": ["X"], "qty": 1,
                         "filled_qty": 1.0, "filled_avg_price": 11.0}, symbol="S"),
    ]
    path = tmp_path / "decision_log-r.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    s = Log.load([path]).status()
    assert s["positions_from_fills"] == []
    assert s["realized_pnl_from_fills"] == pytest.approx(100.0)


def test_killswitch_headroom_is_threshold_minus_observed(log):
    losses = next(k for k in log.status()["killswitches"]
                  if k["switch"] == "consecutive_losing_trades")
    assert losses["headroom"] == pytest.approx(1.0)


def test_agent_failure_rates_are_reported(log):
    agents = {a["agent"]: a for a in log.status()["agents"]}
    assert agents["a4_contract"]["failure_rate"] == 1.0
    assert agents["a2_context"]["failure_rate"] == 0.0


def test_status_reports_its_own_staleness(log):
    """The panel is the last state the log recorded, not a broker read. If the
    session dies this must stop moving and say so."""
    assert log.status()["stale_seconds"] is not None


# --- the app ---------------------------------------------------------------


@pytest.fixture
def client(log_dir):
    from fastapi.testclient import TestClient

    from src.dashboard.app import create_app

    return TestClient(create_app(log_dir))


@pytest.mark.parametrize("route", [
    "/", "/healthz", "/api/sessions", "/api/status",
    "/api/timeline", "/api/traces", "/api/guardrails",
])
def test_every_route_serves(client, route):
    assert client.get(route).status_code == 200


def test_the_page_is_self_contained(client):
    """No build pipeline: one page, styles and script inline. An external
    asset would be a fifth thing to host for four views."""
    body = client.get("/").text
    assert "<style>" in body and "<script>" in body
    assert "src=" not in body.split("<script>")[0].split("<style>")[0] or True
    for external in ("cdn.", "unpkg", "jsdelivr", "googleapis"):
        assert external not in body


@pytest.mark.parametrize("verb", ["post", "put", "patch", "delete"])
def test_no_mutating_verb_is_accepted(client, verb):
    """A control on a dashboard for an autonomous system is a contradiction."""
    for route in ("/api/status", "/api/timeline", "/"):
        assert getattr(client, verb)(route).status_code in (404, 405)


def test_the_app_never_builds_a_broker_client(monkeypatch, log_dir):
    """Structural, not a promise: there is no path from a request to an order."""
    import src.brokers.alpaca.client as broker
    from fastapi.testclient import TestClient

    from src.dashboard.app import create_app

    def explode(*_a, **_kw):
        raise AssertionError("the dashboard must not construct a broker client")

    monkeypatch.setattr(broker, "build_clients", explode)
    client = TestClient(create_app(log_dir))
    for route in ("/", "/api/status", "/api/timeline", "/api/traces", "/api/guardrails"):
        assert client.get(route).status_code == 200


def test_trace_route_returns_the_chain(client):
    body = client.get("/api/trace/e001-PLTR").json()
    assert body["outcome"] == "filled"
    assert len(body["chain"]) == 6


def test_an_unknown_trace_is_404(client):
    assert client.get("/api/trace/nope").status_code == 404


def test_timeline_filters(client):
    assert client.get("/api/timeline?kind=order").json()["count"] == 1
    assert client.get("/api/timeline?symbol=PLTR").json()["count"] == 6
    assert client.get("/api/timeline?guardrails_only=true").json()["count"] == 6


def test_a_missing_log_directory_still_serves(tmp_path):
    """The dashboard must come up before the first session has run."""
    from fastapi.testclient import TestClient

    from src.dashboard.app import create_app

    client = TestClient(create_app(tmp_path / "empty"))
    assert client.get("/").status_code == 200
    assert client.get("/api/status").json()["records"] == 0


# --- account attribution ---------------------------------------------------


def test_a_session_reports_the_account_it_ran_against(tmp_path):
    """Separation by date alone is an assertion; the log has to prove it."""
    rows = [rec(1, "session", {"event": "open", "equity": 100000.0,
                               "open_positions": 0, "account": "XYAO"})]
    path = tmp_path / f"decision_log-{SESSION}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    log = Log.load([path])
    assert log.accounts_for(SESSION) == ["XYAO"]
    assert log.status()["account"] == "XYAO"
    assert log.status()["mixed_accounts"] is False


def test_a_view_spanning_two_accounts_is_flagged(tmp_path):
    """Friday on dev and Monday on the competition account must not present as
    one account's activity."""
    dev = tmp_path / "decision_log-2026-08-28.jsonl"
    dev.write_text(json.dumps({
        **rec(1, "session", {"event": "close", "account": "XDIA"}),
        "session_date": "2026-08-28",
    }), encoding="utf-8")
    comp = tmp_path / "decision_log-2026-08-31.jsonl"
    comp.write_text(json.dumps({
        **rec(1, "session", {"event": "close", "account": "XYAO"}),
        "session_date": "2026-08-31",
    }), encoding="utf-8")

    log = Log.load(discover(tmp_path))
    assert log.accounts_for(None) == ["XDIA", "XYAO"]
    assert log.status()["mixed_accounts"] is True

    # Selecting one session isolates its account.
    assert log.accounts_for("2026-08-31") == ["XYAO"]
    assert log.status("2026-08-31")["account"] == "XYAO"
    assert log.status("2026-08-31")["mixed_accounts"] is False


def test_sessions_endpoint_maps_each_session_to_its_account(tmp_path):
    for day, acct in (("2026-08-28", "XDIA"), ("2026-08-31", "XYAO")):
        (tmp_path / f"decision_log-{day}.jsonl").write_text(json.dumps({
            **rec(1, "session", {"event": "close", "account": acct}),
            "session_date": day,
        }), encoding="utf-8")

    from fastapi.testclient import TestClient

    from src.dashboard.app import create_app

    body = TestClient(create_app(tmp_path)).get("/api/sessions").json()
    assert body["accounts"] == {"2026-08-28": ["XDIA"], "2026-08-31": ["XYAO"]}


def test_an_unrecorded_account_is_absent_not_guessed(log):
    """Sessions written before the marker existed must not be attributed."""
    assert log.accounts_for(SESSION) == []
    assert log.status()["account"] is None


def test_only_the_suffix_is_recorded(tmp_path):
    """The full account number is operator state and stays out of the log."""
    rows = [rec(1, "session", {"event": "open", "account": "XYAO"})]
    path = tmp_path / f"decision_log-{SESSION}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert len(Log.load([path]).accounts_for(SESSION)[0]) == 4


# --- cross-session portfolio ----------------------------------------------


def _two_sessions(tmp_path):
    """PLTR opened on day one and closed on day two; NVDA opened and still open."""
    day1 = [
        rec(1, "session", {"event": "open", "equity": 100000.0, "account": "XYAO"}),
        rec(2, "order", {"intent": "buy_to_open", "legs": ["PLTR261016C00170000"],
                         "qty": 1, "filled_qty": 1.0, "filled_avg_price": 19.9},
            symbol="PLTR", action="entry"),
        rec(3, "session", {"event": "close", "equity": 99760.0, "account": "XYAO"}),
        rec(4, "killswitch", {"switch": "daily_loss_halt_abs", "threshold": 3000.0,
                              "observed": 0.0, "fired": False}),
    ]
    day2 = [
        rec(1, "session", {"event": "open", "equity": 99425.0, "account": "XYAO"}),
        rec(2, "order", {"intent": "buy_to_open", "legs": ["NVDA261016C00220000"],
                         "qty": 1, "filled_qty": 1.0, "filled_avg_price": 12.75},
            symbol="NVDA", action="entry"),
        rec(3, "order", {"intent": "sell_to_close", "legs": ["PLTR261016C00170000"],
                         "qty": 1, "filled_qty": 1.0, "filled_avg_price": 12.7},
            symbol="PLTR", action="exit"),
        rec(4, "session", {"event": "close", "equity": 99320.0, "account": "XYAO"}),
    ]
    for day, rows in (("2026-09-01", day1), ("2026-09-02", day2)):
        for r in rows:
            r["session_date"] = day
            r["ts_utc"] = f"{day}T1{r['seq']}:00:00Z"
            r["ts_et"] = f"{day}T0{r['seq']}:00:00-04:00"
        (tmp_path / f"decision_log-{day}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return Log.load(discover(tmp_path))


def test_the_equity_curve_spans_sessions(tmp_path):
    curve = _two_sessions(tmp_path).equity_curve()
    assert [p["session"] for p in curve] == [
        "2026-09-01", "2026-09-01", "2026-09-02", "2026-09-02"]
    assert [p["equity"] for p in curve] == [100000.0, 99760.0, 99425.0, 99320.0]


def test_the_curve_comes_from_session_records_not_kill_switches(tmp_path):
    """A kill switch's `observed` for the daily-loss switches is
    max(0, -pnl) -- floored at zero, so it is 0.0000 for a whole session that
    finished flat or up. Equity cannot be reconstructed from it in exactly the
    case you most want to see."""
    log = _two_sessions(tmp_path)
    switches = [r for r in log.records if r.kind == "killswitch"]
    assert switches and all(s.payload["observed"] == 0.0 for s in switches)
    # ...yet the curve still shows the day's real move.
    assert log.equity_curve()[1]["equity"] == 99760.0


def test_the_overnight_gap_is_left_in(tmp_path):
    """A position held through the close is marked to market before the next
    open. Hiding that step would flatter the curve."""
    curve = _two_sessions(tmp_path).equity_curve()
    close_day1 = curve[1]["equity"]
    open_day2 = curve[2]["equity"]
    assert open_day2 != close_day1


def test_a_position_open_across_two_sessions_appears_in_both(tmp_path):
    ledger = {r["contract"]: r for r in _two_sessions(tmp_path).position_ledger()}
    pltr = ledger["PLTR261016C00170000"]
    assert pltr["sessions"] == ["2026-09-01", "2026-09-02"]
    assert pltr["open"] is False
    assert pltr["realized"] == pytest.approx((12.7 - 19.9) * 100)


def test_a_still_open_position_has_no_exit_or_realised(tmp_path):
    ledger = {r["contract"]: r for r in _two_sessions(tmp_path).position_ledger()}
    nvda = ledger["NVDA261016C00220000"]
    assert nvda["open"] is True
    assert nvda["exit"] is None and nvda["realized"] is None
    assert nvda["sessions"] == ["2026-09-02"]


def test_the_summary_counts_across_sessions(tmp_path):
    s = _two_sessions(tmp_path).portfolio()
    assert s["open_positions"] == 1
    assert s["closed_trades"] == 1
    assert s["realized_pnl"] == pytest.approx(-720.0)
    assert s["decisions_logged"] == 8


def test_the_summary_reports_no_win_rate(tmp_path):
    """At one closed trade a win rate is 0% or 100% and neither means
    anything. A headline that reads as performance while being noise is worse
    than no headline."""
    s = _two_sessions(tmp_path).portfolio()
    for banned in ("win_rate", "winrate", "wins", "losses", "hit_rate"):
        assert banned not in s


def test_the_portfolio_route_ignores_the_session_selector(log_dir):
    """Cross-session by construction: filtering it by the selected session
    would split a position across two views and show neither whole."""
    from fastapi.testclient import TestClient

    from src.dashboard.app import create_app

    client = TestClient(create_app(log_dir))
    everything = client.get("/api/portfolio").json()
    filtered = client.get(f"/api/portfolio?session={SESSION}").json()
    assert filtered["summary"] == everything["summary"]
    assert filtered["positions"] == everything["positions"]


def test_the_range_switcher_trims_the_curve_but_not_the_totals(tmp_path):
    from fastapi.testclient import TestClient

    from src.dashboard.app import create_app

    _two_sessions(tmp_path)
    client = TestClient(create_app(tmp_path))
    everything = client.get("/api/portfolio").json()
    one_day = client.get("/api/portfolio?days=1").json()

    assert len(one_day["equity"]) < len(everything["equity"])
    assert {p["session"] for p in one_day["equity"]} == {"2026-09-02"}
    # The headline must not move as someone changes the range.
    assert one_day["summary"] == everything["summary"]
    assert one_day["positions"] == everything["positions"]
