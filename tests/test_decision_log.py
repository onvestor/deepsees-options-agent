"""Decision log: shape stability, append-only behaviour, and secret exclusion.

The acceptance test at the bottom is the one that matters: a replayed session
must produce a log that reconstructs every decision, using nothing but the log.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.decisionlog.adapters import (
    agent_call_payload,
    config_fingerprint,
    prefilter_payload,
    signal_action,
    signal_eval_payload,
)
from src.decisionlog.decision_log import (
    DecisionLog,
    Redactor,
    prompt_hash,
    read_records,
    reconstruct_session,
)
from src.decisionlog.schema import (
    SCHEMA_VERSION,
    CapOverridePayload,
    KillSwitchPayload,
    OrderPayload,
    SessionPayload,
    SizingPayload,
    ValidationResult,
    new_trace_id,
)
from src.signals.engine import SignalProfile, SignalSettings, evaluate
from tests.test_signal_engine import SETTINGS, downtrend, flat, uptrend

PROFILE = SignalProfile(9, 2, True, 0.6, "both")

FAKE_ALPACA_KEY = "PKQQQQQQQQQQQQQQQQQQ"
FAKE_ANTHROPIC_KEY = "sk-ant-api03-" + "z" * 40


@pytest.fixture
def logger(tmp_path):
    redactor = Redactor(secrets=[FAKE_ALPACA_KEY, FAKE_ANTHROPIC_KEY], enabled=True)
    instance = DecisionLog(tmp_path / "decision_log.jsonl", redactor=redactor)
    yield instance
    instance.close()


class Candidate:
    """Minimal stand-in for the prefilter's candidate objects."""

    def __init__(self, symbol, failures=()):
        self.symbol = symbol
        self.failures = tuple(failures)


# --- record shape ----------------------------------------------------------


def test_every_record_carries_the_stable_envelope(logger):
    logger.write(SessionPayload(event="open", equity=100_000.0), action="session_opened")
    record = read_records(logger.path, strict=True)[0]

    for field in (
        "schema_version", "record_id", "seq", "ts_utc", "ts_et",
        "session_date", "kind", "action", "reasons", "payload",
    ):
        assert field in record, f"{field} must be present on every record"
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["kind"] == "session"
    assert record["ts_utc"].endswith("Z")


def test_kind_is_derived_from_the_payload_and_cannot_drift(logger):
    logger.write(KillSwitchPayload(switch="daily_loss", threshold=0.03, observed=0.04,
                                   fired=True), action="halted")
    record = read_records(logger.path, strict=True)[0]
    assert record["kind"] == record["payload"]["kind"] == "killswitch"


def test_one_line_per_record(logger):
    for index in range(5):
        logger.write(SessionPayload(event="open", notes=f"line\nbreak {index}"),
                     action="session_opened")
    lines = logger.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    for line in lines:
        json.loads(line)


def test_unknown_payload_fields_are_rejected():
    with pytest.raises(Exception):
        SessionPayload(event="open", not_a_field=1)  # type: ignore[call-arg]


def test_records_are_frozen():
    payload = SessionPayload(event="open")
    with pytest.raises(Exception):
        payload.event = "close"  # type: ignore[misc]


# --- append-only -----------------------------------------------------------


def test_writes_append_and_never_truncate(tmp_path):
    path = tmp_path / "log.jsonl"
    first = DecisionLog(path)
    first.write(SessionPayload(event="open"), action="session_opened")
    first.close()

    second = DecisionLog(path)
    second.write(SessionPayload(event="close"), action="session_closed")
    second.close()

    records = read_records(path, strict=True)
    assert len(records) == 2
    assert [r["action"] for r in records] == ["session_opened", "session_closed"]


def test_sequence_continues_across_a_restart(tmp_path):
    path = tmp_path / "log.jsonl"
    first = DecisionLog(path)
    for _ in range(3):
        first.write(SessionPayload(event="open"), action="session_opened")
    first.close()

    resumed = DecisionLog(path)
    record = resumed.write(SessionPayload(event="close"), action="session_closed")
    resumed.close()

    assert record.seq == 4
    assert [r["seq"] for r in read_records(path, strict=True)] == [1, 2, 3, 4]


def test_a_truncated_final_line_does_not_lose_the_rest(tmp_path):
    path = tmp_path / "log.jsonl"
    logger = DecisionLog(path)
    for _ in range(3):
        logger.write(SessionPayload(event="open"), action="session_opened")
    logger.close()

    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"seq": 4, "trunca')      # process died mid-write

    assert len(read_records(path)) == 3
    with pytest.raises(json.JSONDecodeError):
        read_records(path, strict=True)


def test_records_written_counter(logger):
    for _ in range(4):
        logger.write(SessionPayload(event="open"), action="session_opened")
    assert logger.records_written == 4


# --- nothing secret --------------------------------------------------------


def test_there_is_no_field_for_prompt_text():
    """Structural, not a runtime check: the prompt cannot be logged."""
    from src.decisionlog.schema import AgentCallPayload

    fields = set(AgentCallPayload.model_fields)
    assert "prompt" not in fields
    assert "prompt_text" not in fields
    assert "messages" not in fields
    assert "prompt_hash" in fields


def test_prompt_is_hashed_not_stored(logger):
    secret_prompt = "You are Agent 1. The proprietary regime heuristic is: buy when X."
    payload = agent_call_payload(
        agent="a1_regime", model="claude-sonnet-5",
        rendered_prompt=secret_prompt, response_raw='{"regime":"trending_up"}',
        validation=ValidationResult(status="ok"),
    )
    logger.write(payload, action="agent_ok", symbol="SPY")

    raw = logger.path.read_text(encoding="utf-8")
    assert "proprietary regime heuristic" not in raw
    assert "Agent 1" not in raw
    assert prompt_hash(secret_prompt) in raw
    assert json.loads(raw)["payload"]["prompt_chars"] == len(secret_prompt)


def test_credentials_in_a_model_response_are_scrubbed(logger):
    leaked = f'{{"note":"my key is {FAKE_ANTHROPIC_KEY} ok"}}'
    payload = agent_call_payload(
        agent="a2_context", model="claude-sonnet-5",
        rendered_prompt="p", response_raw=leaked,
        validation=ValidationResult(status="ok"),
    )
    logger.write(payload, action="agent_ok")
    raw = logger.path.read_text(encoding="utf-8")
    assert FAKE_ANTHROPIC_KEY not in raw
    assert "<redacted>" in raw


def test_credentials_in_a_broker_error_are_scrubbed(logger):
    logger.write(
        OrderPayload(intent="buy_to_open", qty=1,
                     broker_error=f"401 unauthorized for key {FAKE_ALPACA_KEY}"),
        action="order_rejected",
    )
    raw = logger.path.read_text(encoding="utf-8")
    assert FAKE_ALPACA_KEY not in raw


def test_credential_shaped_strings_are_scrubbed_even_if_never_seen():
    """A key this process has never held must still not survive."""
    unseen = "sk-ant-api03-" + "q" * 60
    assert unseen not in Redactor(secrets=[], enabled=True).scrub_text(f"oops {unseen}")


def test_high_entropy_catch_all_scrubs_a_mixed_case_key():
    key = "AbCd1234" * 6                                  # 48 chars, upper+lower+digit
    assert key not in Redactor(secrets=[], enabled=True).scrub_text(f"token {key}")


def test_the_catch_all_does_not_eat_hashes(logger):
    """Regression: a sha256 digest is 64 chars of [a-f0-9] and looks like a key.

    An earlier redactor scrubbed prompt_hash itself, destroying the one field
    that makes prompts auditable without disclosing them. Two guards now stop
    that -- the pattern requires mixed case, and digest fields skip heuristics.
    """
    digest = prompt_hash("anything")
    redactor = Redactor(secrets=[], enabled=True)

    assert digest in redactor.scrub_text(f"hash is {digest}")
    assert redactor.scrub({"prompt_hash": digest})["prompt_hash"] == digest
    assert redactor.scrub({"config_fingerprint": digest[:16]})["config_fingerprint"] == digest[:16]


def test_literal_secrets_are_scrubbed_even_inside_a_hash_field():
    """Opting out of heuristics must not opt out of known-secret replacement."""
    redactor = Redactor(secrets=[FAKE_ALPACA_KEY], enabled=True)
    assert FAKE_ALPACA_KEY not in redactor.scrub({"order_id": f"ord-{FAKE_ALPACA_KEY}"})["order_id"]


def test_redactor_walks_nested_structures():
    redactor = Redactor(secrets=[FAKE_ALPACA_KEY], enabled=True)
    scrubbed = redactor.scrub({"a": [{"b": f"x {FAKE_ALPACA_KEY} y"}]})
    assert FAKE_ALPACA_KEY not in json.dumps(scrubbed)


def test_config_fingerprint_hides_values_but_detects_change():
    one = config_fingerprint({"stop_pct": -35.0})
    same = config_fingerprint({"stop_pct": -35.0})
    other = config_fingerprint({"stop_pct": -20.0})
    assert one == same != other
    assert "35" not in one


# --- deterministic decisions -----------------------------------------------


def test_signal_evaluation_logs_every_gate(logger):
    bars = uptrend()
    evaluation = evaluate(bars, PROFILE, SETTINGS)
    logger.write(
        signal_eval_payload(evaluation, PROFILE, str(bars.index[-1]), len(bars)),
        action=signal_action(evaluation), symbol="SPY",
    )
    payload = read_records(logger.path, strict=True)[0]["payload"]
    assert payload["gates"] == dict(evaluation.gates)
    assert set(payload["gates"]) == {
        "confirmation", "atr_displacement", "vwap_alignment", "rsi_guard", "direction_allowed",
    }
    assert payload["triggered"] is True


def test_a_suppressed_signal_is_logged_as_fully_as_a_triggered_one(logger):
    bars = flat()
    evaluation = evaluate(bars, PROFILE, SETTINGS)
    logger.write(
        signal_eval_payload(evaluation, PROFILE, str(bars.index[-1]), len(bars)),
        action=signal_action(evaluation), symbol="SPY", reasons=list(evaluation.reasons),
    )
    record = read_records(logger.path, strict=True)[0]
    assert record["action"] == "signal_suppressed"
    assert record["payload"]["gates"]["atr_displacement"] is False
    assert record["reasons"]


def test_prefilter_multi_label_survives_the_round_trip(logger):
    candidates = [
        Candidate("A", ["spread", "open interest"]),
        Candidate("B", ["spread"]),
        Candidate("C", []),
    ]
    logger.write(
        prefilter_payload(candidates, thresholds={"max_spread_pct_of_mid": 0.06}),
        action="prefilter_complete", symbol="SPY",
    )
    payload = read_records(logger.path, strict=True)[0]["payload"]
    assert payload["total_contracts"] == 3
    assert payload["survivors"] == 1
    assert payload["reason_counts"] == {"spread": 2, "open interest": 1}
    assert payload["sole_reason"] == {"spread": 1}
    assert payload["rejections"]["A"] == ["spread", "open interest"]
    assert sum(payload["reason_counts"].values()) > payload["rejected"]


def test_prefilter_aggregate_detail_omits_per_contract_rows():
    payload = prefilter_payload([Candidate("A", ["spread"])], thresholds={}, detail="aggregate")
    assert payload.rejections == {}
    assert payload.reason_counts == {"spread": 1}


def test_cap_override_records_both_values(logger):
    logger.write(
        CapOverridePayload(cap_name="max_contracts_per_trade", requested=12, cap_value=5,
                           applied=5, stage="sizing"),
        action="size_reduced", symbol="NVDA",
    )
    payload = read_records(logger.path, strict=True)[0]["payload"]
    assert payload["requested"] == 12 and payload["applied"] == 5


def test_killswitch_fire_is_logged(logger):
    logger.write(
        KillSwitchPayload(switch="daily_loss_halt_pct", threshold=0.03, observed=0.041,
                          fired=True, scope="account"),
        action="halted", reasons=["daily loss exceeded"],
    )
    record = read_records(logger.path, strict=True)[0]
    assert record["payload"]["fired"] is True
    assert record["action"] == "halted"


def test_agent_validation_failure_is_logged_with_the_action_taken(logger):
    logger.write(
        agent_call_payload(
            agent="a4_contract", model="claude-sonnet-5", rendered_prompt="p",
            response_raw="not json at all",
            validation=ValidationResult(status="failed", attempt=2,
                                        errors=["JSONDecodeError: line 1"]),
        ),
        action="no_trade", reasons=["schema validation failed twice"], symbol="SPY",
        latency_ms=812.4,
    )
    record = read_records(logger.path, strict=True)[0]
    assert record["payload"]["validation"]["status"] == "failed"
    assert record["payload"]["validation"]["attempt"] == 2
    assert record["action"] == "no_trade"
    assert record["latency_ms"] == 812.4


def test_clamps_are_recorded_with_both_values(logger):
    logger.write(
        agent_call_payload(
            agent="a3_risk", model="claude-sonnet-5", rendered_prompt="p",
            response_raw='{"size_multiplier": 2.5}',
            response_parsed={"size_multiplier": 1.0},
            validation=ValidationResult(
                status="clamped",
                clamps=[{"field": "size_multiplier", "from": 2.5, "to": 1.0, "bound": "max"}],
            ),
        ),
        action="size_clamped", symbol="AMD",
    )
    clamp = read_records(logger.path, strict=True)[0]["payload"]["validation"]["clamps"][0]
    assert clamp["from"] == 2.5 and clamp["to"] == 1.0


def test_long_responses_are_truncated_and_flagged(logger):
    payload = agent_call_payload(
        agent="a6_review", model="claude-sonnet-5", rendered_prompt="p",
        response_raw="x" * 5000, validation=ValidationResult(status="ok"),
        max_response_chars=100,
    )
    assert payload.response_truncated is True
    assert len(payload.response_raw or "") == 100


# --- acceptance: replay reconstructs every decision -------------------------


def replay_session(logger) -> dict[str, int]:
    """Drive a full synthetic session through the log, orders stubbed.

    Returns the ground truth we expect to recover from the file alone.
    """
    start = datetime(2026, 8, 24, 13, 45, tzinfo=timezone.utc)
    clock = iter(start + timedelta(seconds=30 * i) for i in range(200)).__next__
    expected = {"signal_eval": 0, "prefilter": 0, "agent_call": 0, "sizing": 0,
                "cap_override": 0, "killswitch": 0, "order": 0, "session": 0}

    logger.write(SessionPayload(event="replay_start", equity=100_000.0,
                                config_fingerprint=config_fingerprint({"stop_pct": -35.0})),
                 action="session_opened", at=clock())
    expected["session"] += 1

    scenarios = [("SPY", uptrend()), ("NVDA", downtrend()), ("AAPL", flat())]
    for symbol, bars in scenarios:
        trace = new_trace_id()
        evaluation = evaluate(bars, PROFILE, SETTINGS)
        logger.write(signal_eval_payload(evaluation, PROFILE, str(bars.index[-1]), len(bars)),
                     action=signal_action(evaluation), symbol=symbol, trace_id=trace,
                     reasons=list(evaluation.reasons), at=clock())
        expected["signal_eval"] += 1
        if not evaluation.triggered:
            continue

        candidates = [Candidate(f"{symbol}260828C00100000", []),
                      Candidate(f"{symbol}260828C00110000", ["spread", "delta band"]),
                      Candidate(f"{symbol}260828C00120000", ["open interest"])]
        logger.write(prefilter_payload(candidates, thresholds={"min_bid": 0.30}),
                     action="prefilter_complete", symbol=symbol, trace_id=trace, at=clock())
        expected["prefilter"] += 1

        logger.write(
            agent_call_payload(agent="a3_risk", model="claude-sonnet-5",
                               rendered_prompt=f"secret prompt for {symbol}",
                               response_raw='{"size_multiplier":0.6}',
                               response_parsed={"size_multiplier": 0.6},
                               validation=ValidationResult(status="ok")),
            action="size_scaled", symbol=symbol, trace_id=trace, latency_ms=640.0, at=clock())
        expected["agent_call"] += 1

        logger.write(SizingPayload(sizing_capital=100_000.0, capital_source="options_buying_power",
                                   risk_per_trade=1000.0, premium_per_contract=184.0,
                                   base_contracts=5, model_multiplier=0.6, final_contracts=3),
                     action="size_computed", symbol=symbol, trace_id=trace, at=clock())
        expected["sizing"] += 1

        if symbol == "NVDA":
            logger.write(CapOverridePayload(cap_name="max_positions_per_symbol", requested=3,
                                            cap_value=1, applied=1, stage="entry"),
                         action="size_reduced", symbol=symbol, trace_id=trace, at=clock())
            expected["cap_override"] += 1

        logger.write(OrderPayload(intent="buy_to_open", legs=[f"{symbol}260828C00100000"],
                                  qty=1, limit_price=1.89, order_id=f"ord-{symbol}",
                                  status="filled", filled_qty=1, filled_avg_price=1.87,
                                  session_dte=4),
                     action="order_filled", symbol=symbol, trace_id=trace, at=clock())
        expected["order"] += 1

    logger.write(KillSwitchPayload(switch="consecutive_losing_trades", threshold=3, observed=1,
                                   fired=False), action="checked", at=clock())
    expected["killswitch"] += 1
    logger.write(SessionPayload(event="replay_end", equity=99_400.0, open_positions=0),
                 action="session_closed", at=clock())
    expected["session"] += 1
    return expected


def test_a_replayed_session_reconstructs_every_decision(logger):
    """Acceptance criterion, executable.

    Everything asserted below is derived from the file on disk. Nothing
    consults the objects that produced it.
    """
    expected = replay_session(logger)
    logger.close()

    records = read_records(logger.path, strict=True)
    summary = reconstruct_session(records)

    assert summary["records"] == sum(expected.values())
    assert summary["by_kind"] == {k: v for k, v in sorted(expected.items()) if v}
    assert summary["sequence_is_contiguous"] is True
    assert summary["schema_versions"] == [SCHEMA_VERSION]
    assert summary["sessions"] == ["2026-08-24"]
    assert summary["symbols"] == ["AAPL", "NVDA", "SPY"]

    # Two of three setups traded; the flat one was suppressed and is explained.
    assert summary["signals_evaluated"] == 3
    assert summary["signals_triggered"] == 2
    assert summary["signal_gate_failures"]["atr_displacement"] == 1
    assert summary["orders_filled"] == 2
    assert summary["cap_overrides"] == 1
    assert summary["killswitch_fires"] == 0
    assert summary["agent_validation"] == {"ok": 2}
    assert summary["agent_latency_ms"]["mean"] == 640.0
    assert summary["traces"] == 3

    # Every decision for one candidate trade is recoverable by trace id alone.
    spy_trace = next(r["trace_id"] for r in records if r["symbol"] == "SPY" and r["trace_id"])
    chain = [r["kind"] for r in records if r.get("trace_id") == spy_trace]
    assert chain == ["signal_eval", "prefilter", "agent_call", "sizing", "order"]

    # And no prompt text made it into the artifact.
    raw = logger.path.read_text(encoding="utf-8")
    assert "secret prompt" not in raw


def test_reconstruction_needs_no_config_or_network(logger, monkeypatch):
    """The log must be readable with the config directory gone."""
    replay_session(logger)
    logger.close()
    monkeypatch.setenv("DEEPSEES_CONFIG_DIR", "/nonexistent")
    summary = reconstruct_session(read_records(logger.path, strict=True))
    assert summary["records"] > 0


def test_from_config_rotates_daily(tmp_path, monkeypatch):
    from datetime import date

    from src.config import load_config

    monkeypatch.setenv("DEEPSEES_LOG_DIR", str(tmp_path))
    instance = DecisionLog.from_config(load_config(), session_date=date(2026, 8, 24))
    instance.write(SessionPayload(event="open"), action="session_opened")
    instance.close()
    assert instance.path.name == "decision_log-2026-08-24.jsonl"
    assert instance.path.exists()
