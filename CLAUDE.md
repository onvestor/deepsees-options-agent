# DeepSees Options Agent

Autonomous multi-agent options trading system for the Alpaca AI Trading Agents Hackathon
(28 Aug – 4 Sep 2026). Trades intraday debit structures on a curated liquid universe against
Alpaca paper trading.

---

## Governing principle

> **LLMs make judgments. Code makes calculations and enforces limits.**

Agents receive numbers computed deterministically and return *structured intent* — never a
price, never a quantity, never an order. A validation layer sits between every agent and the
broker.

Three invariants, enforced in code, that must hold regardless of model output:

1. **No model output reaches the broker directly.** Agents emit JSON; the order builder
   constructs the actual request.
2. **The risk layer is monotone.** Models can only make positions smaller, stops tighter, and
   exits earlier. There must be no code path by which a model increases exposure.
3. **Fail closed.** Any schema violation, timeout, or ambiguity results in *no trade*, never a
   guess.

If a change would violate one of these, stop and flag it rather than implementing it.

---

## Non-negotiable repo rules

This repo is private now and **becomes public before submission**. Flipping visibility exposes
the entire commit history, not just the current tree. Therefore:

- **Never commit** secrets, prompt text, tuned thresholds, the symbol universe, or decision logs.
- `prompts/`, `logs/`, `*.jsonl`, `.env`, and `.claude/` are gitignored. `config/` is ignored
  **except `*.example.yaml`, which must contain placeholders only**. Add no other exceptions.
- Committed `*.example.*` files carry **placeholder** values only — never real tuned ones.
- **"Prompt text" means both kinds.** The runtime agent prompts under `prompts/` are the IP.
  The *build* prompts — what gets typed into Claude Code while developing — are equally
  sensitive and live in Claude Code's own session storage outside the repo. Keep them there.
  `.claude/` is ignored wholesale so that a build instruction saved as a custom slash command
  cannot become a commit.
- **Commit messages describe the change, never the prompt that generated it.** "add chain
  prefilter and metrics" — not a restatement of the instruction that produced it.
- If a value would change trading behaviour, it belongs in `config/`, not in source.
- Source files read config through `src/config.py`. No magic numbers inline, ever.
- `config/` and `prompts/` are **operator-supplied**. A fresh clone does not contain them and
  cannot trade until they are filled in. That is the IP boundary and it is intentional — but a
  missing file must fail with a message naming the file and its `.example`, never a stack
  trace, and the README must state the boundary plainly so a judge who clones the repo reads
  it as a design decision rather than a broken project.

Before any commit, verify no file under `prompts/`, `logs/`, or `config/` — other than
`config/*.example.yaml` — is staged.

---

## Stack

- Python 3.11+, `alpaca-py`, pandas, numpy, pydantic v2, pytest
- Anthropic SDK for agent calls
- FastAPI + a single-page dashboard (later phase)
- Target deployment: small Linux VM. Local dev is fine; the trading week runs hosted.

Environment variables (see `.env.example`):

```
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ANTHROPIC_API_KEY=
```

---

## Accounts

Two Alpaca paper accounts, under **separate logins**:

| Account | Role | Rule |
| --- | --- | --- |
| `PA3RQB0ZXDIA` | **Dev.** All building, testing, replay, and every round trip before the event. | Use freely. Reset it whenever a clean $100k is wanted. |
| `PA3KLBQXXYAO` | **Competition.** | **Untouched until 28 Aug.** Do not point `.env` at it, do not read it, do not place a single order in it before the event. |

The account number is a **submission field, not a runtime value**. It appears
nowhere in source, config, or the committed history, and nothing branches on
it — `account_summary()` reads `account.account_number` back from the live API
purely for display. Switching accounts on 28 Aug is therefore a **key change
only**: swap `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in `.env`, change
nothing else. If a code change is ever needed to switch accounts, that is a
bug in the config boundary.

Note that the Step 1 commit message (`8249cd3`) quotes account `PA3KLBQXXYAO`
from a dry run made before the split was decided. The commit is not amended —
rewriting history to fix a number in a message is not worth it — but the
write-up must not copy that figure. Dev work belongs to `PA3RQB0ZXDIA`.

## Directory layout

```
src/
  config.py            # loads config/*.yaml + .env; single source of truth
  brokers/alpaca/
    client.py          # auth, session, retry/backoff
    contracts.py       # /v2/options/contracts discovery + pagination
    quotes.py          # option snapshots: quotes + greeks
    orders.py          # single-leg and mleg order construction
    positions.py       # position reads, reconciliation
  signals/             # PURE FUNCTIONS ONLY — no I/O, no network
    indicators.py      # ema, vwap, atr, rsi, percentile helpers
    engine.py          # signal evaluation given bars + a signal profile
  options/
    occ.py             # OCC symbol build/parse
    prefilter.py       # deterministic chain survivor set
    metrics.py         # theta%, gamma, iv_vs_rv, spread cost, breakeven, modeled pnl
  agents/
    schemas.py         # pydantic models for all six agent contracts
    validator.py       # parse, clamp, retry, fail-closed
    runner.py          # generic call wrapper: timeout, logging, fallback
    a1_regime.py  a2_context.py  a3_risk.py
    a4_contract.py  a5_exit.py   a6_review.py
  risk/
    sizing.py          # base size computation (code, pre-model)
    caps.py            # hard caps applied post-model
    killswitch.py      # deterministic halts
  orchestrator/
    session.py         # market-hours state machine
    scheduler.py       # cadence per agent
  decisionlog/         # NOT `logging/` — that shadows the stdlib module
    decision_log.py    # append-only JSONL
cli/                   # cron entry points
tests/
  fixtures/            # recorded Alpaca responses
replay/                # offline bar replay harness
config/                # GITIGNORED
prompts/               # GITIGNORED
```

**`signals/` must stay I/O-free.** Pure functions over dataframes are what make the replay
harness possible, and the replay harness is what lets prompt iteration happen offline instead
of burning live market sessions. There are only five of those.

---

## Alpaca specifics — read before writing broker code

These are verified against Alpaca's docs and are the things most likely to cost hours:

**Options levels.** Paper accounts get Level 3 automatically. Level 3 covers long calls, long
puts, **buy call spread, buy put spread** — debit structures only. Credit verticals are out of
scope. Do not build short-premium paths.

**OCC symbols are unpadded.** Alpaca returns `AAPL240119C00100000` — no space padding of the
root to six characters. Standard OSI padding will likely be rejected. Round-trip a real
returned symbol through `build` and `parse` as a unit test.

**The chain needs two calls, not one.**
- `GET /v2/options/contracts?underlying_symbols=X` returns strike, expiry, type, style,
  `open_interest`, `close_price`. **No bid/ask. No greeks.** Paginated via `page_token`,
  default `limit` 100, and the default `expiration_date_lte` is next weekend — always pass
  explicit date bounds.
- Live quotes and greeks come from the **option snapshots** endpoint. On the Basic plan use
  `feed=indicative`.
- The prefilter's spread and delta filters depend entirely on the second call.

**Cache the two layers differently.** Contract universe: ~15 min. Quotes/greeks: 5–10s. A
single 60s cache is simultaneously too aggressive for the universe and far too stale for entry
decisions.

**Order constraints (`POST /v2/orders`, same endpoint as equities):**
- `qty` must be a whole number; `notional` must not be populated
- `time_in_force` must be `day` or `gtc`; `extended_hours` false or absent
- `type` ∈ {market, limit, stop, stop_limit} — **`stop` and `stop_limit` are single-leg only**
- Multi-leg uses `order_class: "mleg"` with `position_intent` per leg

**There is no broker-side stop on a spread.** Every vertical exit is managed by our own polling
loop sending a closing `mleg` order. Agent 5 and the deterministic exit layer must handle this.

**Expiry behaviour.** ITM contracts auto-exercise if ITM by ≥ $0.01. If buying power is
insufficient, Alpaca sells the position out within an hour before expiry. Never rely on either
as an exit path — the time-stop flat rule exists precisely to avoid them.

**Paper NTAs lag.** Exercise/assignment/expiry activities sync to the Activities endpoint the
*next* day, though balances and positions update instantly. Don't build same-day reconciliation
on NTAs.

**Data feed.** Basic plan: IEX only for equities, indicative only for options. Indicators are
therefore computed on a partial-volume view of the tape. This is a known limitation to disclose
in the write-up, not a bug to chase. Historical option data only exists from Feb 2024.

---

## Agent contracts

Six agents. All outputs are pydantic-validated; all violations clamp or fail closed.
Threshold values live in `config/limits.yaml` — key names are in
`config/limits.example.yaml`.

### Agent 1 — Regime & Signal Profiler
Selects the signal parameterization; does not compute the signal.

```json
{
  "symbol": "NVDA",
  "regime": "trending_up|trending_down|range_bound|choppy|gap_fade|unclear",
  "confidence": 0.0,
  "signal_profile": {
    "ema_fast": 9,
    "confirmation_bars": 2,
    "require_vwap_alignment": true,
    "min_atr_multiple": 0.6,
    "allowed_direction": "long_calls|long_puts|both|none"
  },
  "rationale": "string, <=200 chars"
}
```
Enforced: `ema_fast` and `confirmation_bars` restricted to allowed sets (clamp + log if out of
range); confidence below the configured floor forces `allowed_direction: none`; `regime ==
choppy` forces `none` regardless of model output; profile locked for a configured interval
after emission to prevent bar-to-bar thrash.

### Agent 2 — Context & Eligibility
Runs after a deterministic prefilter. Reads headlines and numeric context; decides tradeability.

```json
{
  "symbol": "NVDA",
  "eligible": true,
  "hard_blocks": [],
  "directional_bias": "bullish|bearish|neutral",
  "bias_strength": 0.0,
  "event_risk": "low|medium|high",
  "iv_assessment": "cheap|fair|rich",
  "notes": "string, <=300 chars"
}
```
Enforced: non-empty `hard_blocks` → ineligible, no override; `event_risk: high` → ineligible;
eligible set truncated to a configured maximum, ranked by `bias_strength`. Earnings proximity
is filtered in code *before* the model is called.

### Agent 3 — Risk Allocator
Base size is computed in code first. The model only scales it down.

```json
{ "size_multiplier": 0.6, "reason": "string, <=200 chars" }
```
Enforced: `size_multiplier` clamped to **[0.0, 1.0]** — shrink or veto only, never enlarge.
Final size = `base_contracts × size_multiplier`, then every hard cap applied on top. Caps always
win; every override is logged. Kill switches are fully deterministic and never consult a model.

### Agent 4 — Contract & Structure Selector
Chooses among a deterministic survivor set, including whether to go single-leg or vertical.

```json
{
  "structure": "single_leg|debit_vertical",
  "primary_symbol": "NVDA260904C00185000",
  "short_symbol": null,
  "expected_hold_hours": 3,
  "reason": "string, <=200 chars",
  "alternate_symbol": "NVDA260904C00190000"
}
```
Enforced: every returned symbol must be in the survivor set; `short_symbol` required and
non-null when `structure == debit_vertical`, and must share expiry with `primary_symbol` and be
further OTM. On any failure, fall back to the deterministic best-ratio survivor as single-leg
and log the fallback.

Structure guidance for the prompt (not enforced in code): low IV rank plus strong regime
confidence favours single-leg for maximum gamma and one spread to cross; elevated IV rank or
moderate conviction favours a vertical. Note that a vertical crosses two bid-asks on entry and
two on exit, which on a short intraday hold frequently exceeds the theta saved.

### Agent 5 — Exit Manager
Deterministic exits are always armed independently. The model may only tighten.

```json
{
  "action": "hold|tighten_stop|scale_out_half|exit_now",
  "new_stop_pct": -25,
  "reason": "string, <=150 chars"
}
```
Enforced: `new_stop_pct` must be strictly tighter than the current stop. Widening, removing, or
adding to a position are **not representable in the schema** and are rejected by the validator
if attempted. This is the monotone-safety invariant — it must be a structural property of the
type, not a runtime check that could be bypassed.

### Agent 6 — Nightly Reviewer
Turns the day's decision log into bounded observations for tomorrow's Agent 1 and 2 prompts.

```json
{
  "observations": [
    {"scope": "AMD|global", "text": "<=200 chars", "expires_after_sessions": 5}
  ]
}
```
Enforced: hard cap on observation count; automatic expiry; observations are injected as
*context only* and can never modify a threshold, cap, or schema constraint.

---

## Failure handling

1. Schema validation fails → one retry with the validation error appended → second failure →
   skip, log, no trade.
2. Timeout on the entry path → skip the trade. Never a partial or best-guess order.
3. Model output conflicts with a hard cap → cap wins, both values logged.
4. Broker call fails → exponential backoff, three attempts, then halt new entries and alert.
5. **Every** agent call is appended to `decision_log.jsonl` with prompt hash, full response,
   latency, validation result, and the action actually taken.

The decision log is the single most valuable artifact this project produces. It backs the demo
video, the write-up, and any judge question about why the agent did something. Build it in the
first session, not the sixth.

---

## Build order and acceptance criteria

Work in this order. Each step has a testable exit condition — don't advance without it.

**0 — Repo hygiene.** `.gitignore` extended, `.env.example` created, directory skeleton in
place, `src/config.py` loading from `config/` with example files present.
*Acceptance:* `git status` clean with `config/` and `prompts/` populated locally.

**1 — Broker round trip.** Connect, fetch a chain for one symbol, pick one contract by a
hardcoded rule, place buy-to-open, read the position back, close it.
*Acceptance:* a full open→close cycle visible in the Alpaca dashboard. No agents, no indicators.
This is the milestone that de-risks everything downstream.

**2 — OCC + contracts + quotes.** Symbol build/parse, paginated contract discovery, snapshot
quotes and greeks, two-tier cache.
*Acceptance:* round-trip test on a live symbol passes; chain fetch returns contracts with
populated bid/ask and delta.

**3 — Signal engine.** EMA/VWAP/ATR/RSI as pure functions; `engine.py` evaluates a signal
profile against a bar frame.
*Acceptance:* unit tests on synthetic frames with known answers; zero network calls in
`src/signals/`.

**4 — Chain prefilter + metrics.** Survivor set plus the six computed metrics.
*Acceptance:* on a live chain, returns a survivor set of plausible size with every metric
populated and no NaNs.

**5 — Risk engine.** Base sizing, hard caps, kill switches.
*Acceptance:* property test — no combination of inputs produces a size exceeding any cap.

**6 — Decision logger.** Append-only JSONL with the full record shape.
*Acceptance:* a replayed session produces a log that reconstructs every decision.

**7 — Agent layer.** Schemas, validator, runner, then the six agents.
*Acceptance:* each agent handles malformed output, timeout, and out-of-range values without
placing a trade. Test with deliberately broken mock responses.

**8 — Replay harness.** Offline bar replay driving the full pipeline with orders stubbed.
*Acceptance:* a full session replays end-to-end without network access to the broker.

**9 — Orchestrator, CLI jobs, dashboard.**

---

## Testing

- Record real Alpaca responses into `tests/fixtures/` and test parsers against those, not
  against hand-written JSON. The shape surprises are the whole point.
- Agent tests use mocked model responses including malformed ones — the fail-closed paths
  matter more than the happy path.
- `src/signals/` and `src/options/metrics.py` are pure and must have direct unit tests.
- Never place live paper orders from the test suite.

---

## Conventions

- Type hints throughout; pydantic for every boundary.
- All times in ET internally; store UTC, convert at the edges.
- Log structured, never `print`.
- Small commits with clear messages. Assume every commit will be read by a judge.
