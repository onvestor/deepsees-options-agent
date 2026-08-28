# DeepSees Options Agent

Autonomous multi-agent **swing** options trading system for the Alpaca AI Trading Agents
Hackathon (28 Aug – 4 Sep 2026). Directional theses on a curated liquid universe, expressed as
**debit structures** rather than shares — long calls, long puts, and debit verticals — against
Alpaca paper trading. The option is leverage on a stock-like directional view, not a
volatility bet.

**Hold horizon: 1–5 trading sessions.** This is a positional system. There is no intraday flat
rule. All numeric values live in `config/limits.yaml`; bands quoted here are starting points to
be set by measurement, not constants to hardcode.

Three design consequences follow from the horizon, and must not drift:

- **Theta matters, but less than friction.** Over a 3-session hold, round-trip spread cost on
  an illiquid contract frequently exceeds theta. Optimise total friction, not theta alone.
- **Vega is a live risk.** Vega scales with time to expiry. A long-dated contract can lose more
  to an IV decline than it gains from a correct directional call — uncompensated risk for a
  directional strategy.
- **Overnight gap risk is real and unhedgeable by polling.** Positions are held through
  sessions we cannot poll. Sizing must assume the stop can be gapped through.

---

## Session handoff — 28 Aug 2026

State at the end of the fourth build session. Read this first.

**The competition window opened today.** Every step is built and the system ran a
live unattended session end to end. What is left is operational, not structural --
see "Before Monday" below, which is the only section that blocks a trading day.

### Build status

| Step | State |
| --- | --- |
| 0 — Repo hygiene | Done. `fedf4d2` |
| 1 — Broker round trip | **VERIFIED 25 Aug**, and re-verified 28 Aug through the real order builder rather than the spike. See "The order path" below. |
| 2 — OCC, contracts, quotes, cache | Done. `7f7dfd6` |
| 3 — Signal engine | Done. `5f424ef`. Timeframe-agnostic. |
| 4 — Prefilter + metrics | Done, then substantially revised: monthly-anchored expiry selection (`13298c8`), expiry-type awareness, spread filter rebuilt (`09868fa`). |
| 5 — Risk engine | Done. `ae38523` |
| 6 — Decision logger | Done. `bd94998`, extended with `AgentOverridePayload` (`31d686b`). |
| 7 — Agent layer | **Done, 28 Aug.** All six prompts written; real Anthropic transport behind the `transport()` callable. |
| 8 — Replay harness | **Done, 28 Aug.** Offline bar replay, synthetic chain, stubbed orders, record-once/replay-many transports. `python -m cli.replay_session`. |
| 9 — Orchestrator | **Done, 28 Aug.** Ran live and unattended: `cli/run_session.py --live`. |
| 9b — Dashboard | **Done, 28 Aug.** Four read-only views over the decision log; `python -m cli.dashboard`. This is the submission demo URL. |

~1,753 offline tests, 10 live. The suite runs with **`config/` absent** — verified by moving
it aside — so a fresh clone can run everything offline.

### Step 7 — what exists and what does not

| Piece | State |
| --- | --- |
| `schemas.py` | Done. `bae776d`. Six contracts; widening and size-increase unrepresentable. |
| `validator.py` | Done. `31d686b`. Parse, clamp, force, retry once, skip. |
| `runner.py` | Done. `8ef4348`. Path-aware timeouts, hashed prompts, restatement backstop, failure-rate counter. |
| `prompt_loader.py` | Done. `$field` substitution, missing field raises. |
| Agent 1 — regime | Done. `77a51ec`. Session-locked. |
| Agent 2 — context | Done. `5df9b15`. Earnings filter ahead of the model. |
| Agent 4 — contract | Done. `6b0dd76`. Survivor-set bound, visible fallback. |
| Agent 3 — risk | Done. `69f4d82`. Shrink only, caps after. |
| Agent 5 — exit | Done. `2a6e84e`. Failure continues; `tightens()` gates every stop. |
| Agent 6 — review | Done. Observations are text only. |
| End-to-end pipeline test | Done. All six wired against stubs. |
| **Prompts** | **NONE. Not one of the six has a prompt.** |

**No agent has a prompt.** Every module loads its prompt by name from the gitignored
`prompts/` and fails with a message naming the file when it is absent. All six are tested
against stub transports, so the wiring is proven independently of prompt quality — which was
the point of doing it in this order. A wiring bug and a prompt bug look identical from
outside; when prompts arrive, any failure is attributable to them.

Prompt files needed, by name: `a1_regime.txt`, `a2_context.txt`, `a3_risk.txt`,
`a4_contract.txt`, `a5_exit.txt`, `a6_review.txt`. Each module's `as_fields()` lists exactly
the `$placeholders` its template may reference; an unknown one raises rather than rendering.

### Entry order manager — designed, not built

Specified in full under "Entry order management" in this file. **No code exists.** Every
threshold is declared in `limits.example.yaml` under `execution:` and left unset, so reading
one raises `ConfigError` naming the key and the manager cannot start half-configured.

The design decisions worth not rediscovering: the step ceiling is derived per contract from
whether the trade is still worth taking at that price, displacement is capacity-driven rather
than staleness-driven, and `worst_current_edge` stays off until the edge model is validated
against realised outcomes.

### Before Monday — the only things that block a trading day

1. **Switch `.env` to the competition keys.** Comment out the active pair, uncomment the
   competition pair. **The `.env` header comment is not evidence:** on 28 Aug it read
   an account number that the live keys did **not** address. Which account a key pair
   addresses is a property of the keys, and only the broker can tell you. Verify with
   `python -m cli.preflight --expect-account <number> --expect-equity 100000 --require-level 3
   --require-flat` — exit 0 or it is not switched.
2. **Re-register the scheduled task from an elevated PowerShell.** It fell back to
   `LogonType=Interactive`, which fires only while the operator is logged on — a machine at the
   lock screen on Monday would simply not trade. `powershell -ExecutionPolicy Bypass -File
   scripts\install_tasks.ps1 -ExpectAccount <number> -TestAt '<local datetime>'` when elevated
   registers `S4U` instead.
3. **Two positions are open** from the 28 Aug session and will be managed from the open.

`DeepSees-Session` fires 05:45 local = **08:45 ET**, Mon–Fri. The machine is Pacific; both
zones share DST dates so the three-hour offset is stable, and the task's local time is the
thing to check if a firing ever looks an hour out. The wrapper gates the session on the
account preflight and **refuses to start** if the keys address the wrong account.

### The first live session — 28 Aug, dev account

1,586 decision-log records, 11:51–16:04 ET. Read it with `python -m cli.session_report`.

| | |
| --- | --- |
| agent calls | 85 across all six, zero failures, zero clamps, median 1.8–4.7 s |
| schema retries | 3, all a trailing comma from Agent 2, all recovered on the retry |
| skips | 102, by stage: entry 50, signal 13, a3 13, a2 12, order 9, a2_earnings 3, a1 2 |
| orders | 11 placed, **2 filled** — the passive mid limit is doing what the fill study said |
| forces | 13, all `agents.a2.min_bias_strength` or the eligible-set truncation |
| kill switches | 1,240 evaluations |

**What the session proves.** The retry path recovered malformed model output in production.
The earnings exclusion blocked NVDA in code before any model call. `max_positions_per_symbol`
refused repeat entries on held symbols. Agent 5 marked both open positions every 30 minutes
and held them against the −35% stop. Agent 6 ran at the close.

**Two bugs it found that no offline test could.**

*The kill switch was reading sizing capital, not equity.* `options_buying_power` falls by the
premium of every position opened, so two open positions showed as a 4,505 "loss" against a real
equity change of −170 and tripped three halts after two trades. It would have halted Monday
around 09:20. Fixed; 12 spurious fires before, **zero across 895 evaluations after**.

*`position.contract_symbol` on a class whose field is `symbol`* took out the exit handler on
every tick. The orchestrator isolated it and the deterministic exits stayed armed, but no open
position was managed until it was caught. It reached a live session because `live.py` had no
tests; `tests/test_live_session.py` now covers it.

**The one calibration finding.** Agent 2 returned `bias_strength` 0.35 for symbols it disliked
and 0.45 for ones it liked, against a floor of 0.40 — so the model clusters either side of the
gate and every rejection shows as a force. Either the prompt needs calibration guidance or the
floor wants moving. The log has the evidence; do not guess.

### The dashboard — the submission demo URL

`python -m cli.dashboard --host 0.0.0.0 --port 8080`. FastAPI plus one self-contained page, no
build step. Four views: session timeline, decision trace, guardrail events, live status.
`--check` renders every route and asserts read-only.

**Every figure comes from `decision_log.jsonl` and nothing else.** No broker calls — if a fact
is not in the log the dashboard cannot show it, which is what makes it an audit trail rather
than a second unverifiable view. The status panel reports its own staleness rather than looking
current after the session stops.

**Read-only is structural**, not a promise: no mutating verb is routed, no broker client is
ever constructed, and a test monkeypatches `build_clients` to raise and asserts every route
still serves. A "close position" button on a dashboard for an autonomous system is a
contradiction.

Safe to expose: prompts are stored as hashes, credentials are scrubbed by the log's redactor,
and account numbers were never written to it.

### The order path — built and proven live, 28 Aug

`src/brokers/alpaca/orders.py` and `positions.py` exist. **Single-leg only** (see the vertical
finding above). `cli/step1_roundtrip.py` no longer carries its own chain fetch, snapshot
batching, candidate filter or order construction — it routes through the real modules, and
dropped from 559 lines to 305.

**Verified on the dev account, 28 Aug**, through the real path rather than the spike:

| | |
| --- | --- |
| contract | `SPY261016C00775000`, chosen by the real prefilter (111 scanned, 8 survivors) |
| entry | passive mid limit 15.48 → filled 15.44 |
| read-back | reconciled from Alpaca; OCC fields parsed from the symbol |
| exit | stepped ladder (15.64, 15.25), high urgency, filled 15.44 on rung 2 |
| result | position closed, book empty, no stray orders |

**The first attempt did not fill and was cancelled at the 20s timeout.** That is the design,
not a failure — the 25 Aug study measured a 55% mid-fill rate on buys — and it is why the
entry order manager exists as a separate concern.

The realised round trip came out at 0.00% of premium against a 2.39% quoted spread, because
the underlying moved between entry and exit. **Do not generalise from it.** That ratio is an
unstable statistic at n=1, which is the same trap the Step 1 round trip's apparent 8–10x set —
see "Realised friction" below.

**Two shapes, on purpose.** Entry is one passive mid limit that fills or does not; repricing
belongs to the entry order manager on its own clock. Exit is a stepped limit whose urgency is
keyed to the **exit reason** — stop, expiry-week and agent-exit are HIGH; max-hold is MEDIUM;
target and scale-out are LOW. Every ladder terminates at the bid, so an exit that must happen
happens; urgency changes only how fast. An unrecognised reason maps to HIGH: failing closed in
the direction of leaving.

**`positions.py` never caches.** Every read hits Alpaca. The orchestrator's `reconcile` job is
only worth running if it is authoritative, the cancellation race corrupts exactly this state,
and assignment/exercise/expiry change the book with no order of ours involved.

**Still unbuilt:** the entry order manager (design under "Entry order management"), and the
dashboard.

### Remaining work, in order

1. **The three items under "Before Monday" above.** Nothing else blocks a trading day.
2. **The entry order manager** (design further down). `execution.entry_reprice_cadence_seconds`
   is still deliberately unset. 9 of 11 orders on 28 Aug did not fill at the mid, which is what
   this component exists to improve — it is the highest-value remaining build.
3. **Prompt iteration** against the replay harness. Record once with
   `python -m cli.replay_session --record`, then iterate offline for free — a recording is
   keyed by the rendered prompt's hash, so an edited prompt is a miss that names the agent
   rather than a stale hit.
4. **Re-measure `monthly_min_sessions`.** Still 21, still selecting a 34–38 session expiry.
   The open question below is unchanged.
5. **Dashboard.** FastAPI over the decision log.

### Two open threshold questions

**1. `monthly_min_sessions` at 21 is pushing realised DTE long.** On 25 Aug it selected the
October monthly at **38 sessions / 53 calendar days**, because the September monthly sat at 18
sessions and missed the floor by three. Lowering the floor to **~14** would have admitted
September and roughly halved realised DTE. A nearer monthly is cheaper in premium and vega and
worse in theta; which dominates over a 1–5 session hold is exactly what the bucket sweep
answered for fixed bands and has **not** answered for this rule. Measure before changing.

**2. Does the FMP paid tier lift the 402?** `CRM`, `ORCL`, `QCOM` and `SPCX` all return
**HTTP 402** on the current free tier, so their earnings dates are unavailable, the exclusion
fails closed, and they cannot be traded — regardless of liquidity. ORCL was the best contract
in the whole 48-symbol screen (OI 8,029 at a 1.3% spread) and is unusable. Third-party pricing
puts Starter at **$22/mo** and Premium at **$59/mo** (FMP's own pricing page 403s to automated
fetches, so confirm on the site). **What is not known is which tier lifts the 402 for those
specific symbols** — that is the only question that matters, and it is worth a one-month
Starter trial tested against exactly those four tickers before committing. If it works, it
buys more universe than any threshold change available.

### Accounts

`.env` points at the **dev** account. The competition account stays **untouched until 28 Aug**;
switching is a key change only.

### Things deliberately left undone

- **Verticals: designed, measured, and NOT shipped.** Decided 28 Aug — see
  "Debit verticals are unconstructible at swing DTE" below. `compute_vertical_metrics`
  stays (31 tests); nothing builds pairs, and nothing should until the delta bands and
  the width cap are re-derived together. **Ship single-leg only.**
- **Exit values.** `exits.stop_pct` / `target_pct` still hold intraday-era values (−35 / +60)
  against the revision's −40 / +75 starting points. `exits.max_hold_sessions` and
  `exits.min_sessions_to_expiry` were added 28 Aug (both 5): the block previously held only
  `max_hold_minutes`, so nothing could enforce a hold counted in sessions.
- **`roundtrip.*` config block.** Step 1 spike only. Delete it once the real order builder
  lands — Step 1 does not route through the prefilter, so it never exercised the monthly rule.
- **AMD dropped from the universe** (25 Aug): structurally unaffordable at this delta band,
  and the one name where quoted spread understated realised by 2×. **PLTR added.**
- **Slippage is measured but not applied.** The friction metrics still use quoted spread. The
  study says that is ~15% optimistic at the population level; nothing has been recalibrated.

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

| Role | Rule |
| --- | --- |
| **Dev.** All building, testing, replay, and every round trip before the event. | Use freely. Reset it whenever a clean $100k is wanted. |
| **Competition.** | **Untouched until 28 Aug.** Do not point `.env` at it, do not read it, do not place a single order in it before the event. |

**The account numbers themselves are operator state and live outside the repo** —
in the Alpaca dashboard and on the submission form. They are not in `.env`
either: `.env` holds the key pair, and *which* account that pair addresses is a
property of the keys, never a configured value. Nothing in source or config
branches on an account number;
`account_summary()` reads `account.account_number` back from the live API purely
for display. Switching accounts on 28 Aug is therefore a **key change only**:
swap `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in `.env`, change nothing else. If
a code change is ever needed to switch accounts, that is a bug in the config
boundary.

Earlier commits do contain the two numbers, and this file quoted them until
23 Aug. History is **deliberately not being rewritten**: a paper account number
is not a credential — it is inert without the API keys — and a `filter-repo`
rewrite is a disproportionate response that would invalidate every existing
commit hash for no security gain. The rule going forward is that new work does
not add them back.

## Directory layout

```
src/
  config.py            # loads config/*.yaml + .env; single source of truth
  brokers/alpaca/
    client.py          # auth, session, retry/backoff, sizing_capital guard
    calendar.py        # trading sessions; DTE counted in sessions, not days
    contracts.py       # /v2/options/contracts discovery + pagination
    quotes.py          # option snapshots: quotes + greeks
    orders.py          # single-leg and mleg order construction
    positions.py       # position reads, reconciliation
  signals/             # PURE FUNCTIONS ONLY — no I/O, no network
    indicators.py      # ema, vwap, atr, rsi, percentile helpers
    engine.py          # signal evaluation given bars + a signal profile
  options/
    occ.py             # OCC symbol build/parse (fixed-width, root = symbol[:-15])
    prefilter.py       # deterministic chain survivor set; narrows BEFORE snapshots
    metrics.py         # PURE. six single-leg metrics + vertical risk/reward
  earnings/
    calendar.py        # FMP next-earnings dates; fail-closed cache
  agents/
    schemas.py         # pydantic models for all six agent contracts
    validator.py       # parse, clamp, force, retry once, fail-closed
    runner.py          # call wrapper: path-aware timeout, hashed prompts,
                       #   prompt-restatement backstop, failure-rate counter
    prompt_loader.py   # $field templates from the gitignored prompts/
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
  step1_roundtrip.py   # broker round trip (Step 1)
  monday_measurement.py # DTE band x strike window sweep
tests/
  fixtures/            # recorded Alpaca responses
replay/                # offline bar replay harness
config/                # GITIGNORED
prompts/               # GITIGNORED
cache/                 # GITIGNORED (provider data, refetchable)
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
as an exit path — the "never hold into expiry week" rule exists precisely to avoid them.

**Paper NTAs lag.** Exercise/assignment/expiry activities sync to the Activities endpoint the
*next* day, though balances and positions update instantly. Don't build same-day reconciliation
on NTAs.

**Data feed.** Basic plan: IEX only for equities, indicative only for options. Indicators are
therefore computed on a partial-volume view of the tape. This is a known limitation to disclose
in the write-up, not a bug to chase. Historical option data only exists from Feb 2024.

---

## DTE — measured, not assumed

**Settled 24 Aug: the expiry is chosen, not bounded.** The rule is the nearest standard
monthly expiry at least `prefilter.monthly_min_sessions` trading sessions out. A fixed
calendar-day band was tried first and does not work: a 15-day-wide window inside a ~30-day
monthly cycle misses the monthly roughly half the time. On 24 Aug a 30–45 day band fell
entirely between the September monthly (25 days) and the October one (53 days), and requiring
monthlies inside it produced an **empty survivor set on every symbol**. Anchoring on the
expiry always lands on the liquid contract and can never empty.

The cost is that **realised DTE varies per trade** — roughly `monthly_min_sessions` to
`monthly_min_sessions + 21`. It is therefore a property of the entry, recorded on
`PrefilterResult.target_session_dte` and in the logged thresholds, never assumed from config.
`dte_min`/`dte_max` survive only as a guard envelope that raises if the chosen monthly falls
outside it.

**The knob, if realised DTE runs long.** `monthly_min_sessions` at 21 forced the 24 Aug
selection out to the October monthly at **38 sessions / 53 calendar days**, because the
September monthly sat at 18 sessions and missed the floor by three. Lowering the floor to
**~14** would have admitted September and roughly halved realised DTE. That is the lever to
reach for if DTE keeps landing at the long end — but it is a **measurement for later in the
week, not a guess now**: a nearer monthly is cheaper in premium and vega and worse in theta,
and which dominates over a 1–5 session hold is exactly the question the Monday sweep answered
for buckets and has not yet answered for this rule. Do not change it without the numbers.

**What follows is the bucket measurement that produced the rule above**, kept because it is
the evidence and because the same columns are what any re-measurement must report. The bucket
grid has one blind spot worth remembering: the October monthly the rule now selects sits at 53
days, in the gap between the 30–45 and 60–90 buckets, so the grid never priced the contract
the system actually buys. SPY looked unaffordable on bucket medians and clears comfortably at
the chosen expiry.

Run the comparison on SPY, NVDA and AMD across three buckets — 30–45, 60–90, 120+ days — and
report, for a modelled 3-session hold:

| Metric | Why |
|---|---|
| `theta_pct_per_day` × 3 sessions | the cost longer DTE is meant to avoid |
| `spread_cost_vs_expected_move` round trip | the cost longer DTE adds |
| total friction per unit of delta | the number that actually decides it |
| vega as % of premium | the risk longer DTE introduces |
| premium per contract | see the capital constraint below |
| contracts clearing the liquidity filters | long-dated contracts often fail OI/volume |
| fraction of contracts spanning earnings | see the earnings section |

**The capital constraint is likely to be binding.** At a 5%-of-equity cap on a $100k account,
one position may not exceed $5,000. A 0.70-delta 4-month call on a $700+ underlying can cost
$5,500–6,500 — a single contract that cannot be bought at all, and no sizing granularity even
where it fits. Report how many contracts clear the caps in each bucket; a bucket where the
answer is "one or zero" is not viable regardless of its theta profile.

Expected outcome is that the middle bucket wins, but the measurement decides.

**Hard constraint regardless of the band chosen:** expiry must clear the maximum hold window by
a wide margin. Never manage a position in its final week — theta acceleration and widening
spreads both bite there.

---

## Realised friction — measured 25 Aug, 20 round trips

Every friction metric in `src/options/metrics.py` is computed from the **quoted** spread.
Whether that predicts what a round trip actually costs was an open assumption until it was
measured: 20 round trips, one contract each, across SPY, IWM and AMD on the dev paper account.

| | quoted median | realised median | ratio (median) |
|---|---|---|---|
| SPY | 1.07% | 0.94% | 0.8x |
| IWM | 1.43% | 1.51% | 1.1x |
| AMD | 1.09% | 1.90% | 2.0x |
| **all** | **1.30%** | **1.49%** | **1.0x** |

Realised round trip: mean 1.41%, median 1.49%, range 0.34%–2.81%, sd 0.58% of premium.

**The population conclusion: quoted spread is approximately unbiased.** It understates realised
cost by roughly **15%**, not by an order of magnitude. A modest haircut on the friction metrics
is defensible; anything larger is not supported by this data.

**The per-trade conclusion is different and matters more.** The realised/quoted ratio spans
**0.3x to 8.3x** across twenty trades. Quoted spread predicts *population* cost well and
*individual* cost badly. Never treat a single contract's quoted spread as its expected cost.

**Why the Step 1 round trip looked like 8–10x, and why that was misleading.** Step 1 filled a
contract whose quoted spread was $0.01 — the tightest in any sample here. The ratio is a
fraction with a tiny denominator, so it exploded; its *realised* cost was 0.78%, better than
the median above. The generalisation from that single trade was wrong. **The ratio is an
unstable statistic whenever the quoted spread is small, and should never be reasoned from at
n=1.** Report realised cost in absolute percentage-of-premium terms, which is stable.

**Per-name calibration beats one global factor.** AMD was the outlier that mattered — quoted
understated realised by 2x consistently, worst single trip 2.81%. It has since been dropped
from the universe for being structurally unaffordable at this delta band, but the lesson
stands: a name whose quoted spreads systematically understate is a name to measure before
trusting, not to average into a universe-wide constant.

**Caveats, stated plainly.** These are paper fills against Alpaca's simulator on the indicative
options feed, taken in one session. Direction and magnitude are evidence; they are not a
substitute for the same measurement against a live-money fill engine, and the write-up should
say so.

## Metric acceptance bands — wired 28 Aug, one of six does the work

The six `metrics.*` bands were **declared in config and read by nothing** until 28 Aug. The
prefilter gated on open interest, bid, spread percentage, delta, DTE and expiry type, computed
the metrics, and used them only for ranking and the log. They are now rejection gates behind
`prefilter.apply_metric_bands`.

Measured on nine live chains at 34 sessions DTE, same chain scored both ways:

| band | rejections | verdict |
| --- | --- | --- |
| `max_breakeven_move_pct` | **37 of 48 survivors** at 0.02 | intraday-era; retuned to 0.05 |
| `max_theta_pct_per_day` | 0 | inert at 0.12 |
| `min_gamma_per_1pct` | 0 | inert |
| `max_iv_vs_rv_ratio` | 0 | inert at 1.60 |
| `max_spread_cost_pct_of_premium` | 0 | **cannot ever fire** — see below |
| `min_modeled_pnl_ratio` | 0 | inert at 1.30 |

**`max_breakeven_move_pct: 0.02` was the whole filter, and it was the wrong one.** At 34
sessions DTE a 0.55–0.75 delta call carries enough time value that breakeven sits 1.6–7.0% above
spot. A 2% ceiling kept 15% of survivors — and structurally kept only the index ETFs, because a
volatile single name needs a larger percentage move at the same delta. It emptied NVDA, PLTR,
AAPL, MSFT, TSLA and META entirely. Retuned to **0.05** on the measured distribution (median
2.6%, p75 3.7%, max 7.0%), which keeps 90% and cuts only the far tail; PLTR still empties, which
is the band doing its job rather than the band being wrong.

**`max_spread_cost_pct_of_premium` is redundant by construction.** Premium is the mid, so it is
arithmetically the same quantity as `prefilter.max_spread_pct_of_mid`, which is tighter (0.04)
and runs first. It is kept as a backstop and annotated as one. It rejected zero contracts not
because spreads were tight but because it is unreachable.

**The better long-term gate is ATR-relative.** `breakeven_distance_atr` already exists and is
the comparable-across-names measure, in exactly the way `gamma_per_1pct` is. A percent-of-spot
band systematically discriminates against volatile names; an ATR band does not. Worth measuring
before the write-up.

## Modelled hold — one value, derived

`metrics.modeled_hold_hours` is **gone**. It held 4.0 — an intraday leftover — beside
`metrics.modeled_hold_sessions: 3`, and the two disagreed. The decay term in `compute_metrics`
is `theta × hold_hours / theta_day_hours`, so a 4-hour hold understated theta by a factor of 18
in every modelled P&L the prefilter ranks on. `modeled_hold_hours(limits)` now derives it as
`modeled_hold_sessions × theta_day_hours`, so the two cannot diverge again.

Effect on ranking, checked rather than assumed: the survivor **order is unchanged** — theta is a
near-uniform subtraction across near-the-money strikes — but `modeled_pnl_1atr` was overstated
by roughly 16%, and every modelled figure in the decision log with it.

## Universe screening — 25 Aug, 48 symbols

**Nine sector ETFs returned zero survivors** — XLK, XLV, XLI, XLY, XBI, XOP, IYR, EFA and IJR
— despite each having 21–34 contracts inside the strike window and no earnings risk at all.
They have the strikes and they have the no-print advantage; their options simply do not clear
the delta and liquidity gates. **ETF safety does not generalise past the big three index
funds.** SPY, QQQ and IWM are not representative of "ETFs" as a class, and a sector fund should
never be added on the assumption that it inherits their liquidity. Measure it first.

**Data coverage disqualifies symbols that liquidity would have accepted.** FMP returns HTTP 402
on this plan tier for CRM, ORCL, QCOM and SPCX. Their earnings dates are unavailable, the
earnings rule fails closed, and declaring a single name in `no_earnings` would be a lie of
exactly the kind that section forbids — so they cannot be traded regardless of how good the
contract is. ORCL was the single best contract in the whole screen (OI 8,029 at a 1.3% spread)
and is unusable. **The binding constraint on universe expansion is the earnings feed, not the
prefilter thresholds.**

## Delta bands

| Structure | Leg | Band |
| --- | --- | --- |
| Single leg | — | 0.55–0.75 |
| Debit vertical | Long | 0.55–0.70 |
| Debit vertical | Short | 0.25–0.35, same expiry, further OTM |

Rationale for the prompt, not enforced in code: slightly ITM gives 55–75% participation in the
underlying move with defined risk and modest theta. Past ~0.80 delta you are paying for
intrinsic value and crossing wider spreads for a diminishing gain over simply holding stock.

**Verticals need a viable hold window.** A debit spread converges toward maximum value only near
expiry. On a long-dated contract over a 3-session hold a vertical barely moves — the short leg's
decay offsets the long leg's gain. If the Monday measurement lands on a long DTE band, Agent 4's
structure guidance should favour single-leg heavily.

### Debit verticals are unconstructible at swing DTE — measured 28 Aug

**The two bands in the table above contradict each other at the expiry this system actually
buys.** A 0.55–0.70 long leg and a 0.25–0.35 short leg in the same expiry are further apart in
strike than `agents.a4.vertical_max_width` (15.0) permits, and the gap widens with both DTE and
implied volatility. Measured on a modelled SPY chain at spot 500:

| DTE | min achievable width | pairs inside 2.5–15.0 |
| --- | --- | --- |
| 53 days (**the chosen monthly**, 38 sessions) | 22 | **0 of 192** |
| 25 days (nearer monthly, ~18 sessions) | 15 | 1 of 99 |
| 14 days | 12 | 10 of 48 |

Robust across volatility and **worse as vol rises** — at the chosen monthly, pairs exist only
below ~11% IV; at 16% and above there are none at any width. The one 25-day pair that did
construct captured **13.1%** of max value over a 3-session hold, with modelled P&L of 1.117 per
share against the best single leg's 2.571.

So building the pairing layer would produce a survivor set of zero on nearly every scan, and
Agent 4 would fall back to single-leg on every call. The blocker is **not** implementation
effort — it is that the delta bands and the width cap were sized for the intraday design and
were never re-derived for a 21+ session expiry.

**Before anyone builds this, re-derive the three numbers together:** `vertical_short_delta_*`,
`vertical_max_width`, and `monthly_min_sessions`. A short leg nearer the money, a wider cap, or
a nearer expiry each make pairs constructible; none of them is free, and the capture percentage
above says the nearer expiry is the one that matters. Confirm against a live chain first — the
table is from the replay chain model, whose delta curve is Black-Scholes and whose absolute
widths are therefore indicative rather than exact.

Two further reasons the answer would still be "single leg" even if pairs constructed:
`src/brokers/alpaca/orders.py` does not exist — the only order-placing code is the Step 1 spike,
single-leg — so verticals would add an `mleg` path to a component that has not had its first
one; and there is no broker-side stop on a spread, so a multi-session vertical is unprotected
overnight.

---

## Signal cadence

A 9 EMA cross on 5-minute bars is an intraday trigger and is **no longer appropriate**.

- **Signal evaluation:** daily bars, with hourly as a secondary confirmation timeframe.
- **Agent 1 (regime):** once per session pre-market, not every 30 minutes. Regime for a
  multi-session hold is a daily judgment. The 30-minute profile lock becomes a full-session lock.
- **Agent 2 (context):** once per session pre-market, unchanged in shape.
- **Agent 5 (exit):** every 30 minutes per open position, plus immediately on a ±20% premium
  move. Not every 5 minutes.

This cuts the call budget by roughly an order of magnitude — ~250–400 per session down to
~30–60. Latency tolerances relax accordingly.

`src/signals/` is unaffected: the indicators are timeframe-agnostic pure functions. Only the bar
frame passed in and the cadence of evaluation change.

---

## Entry order management — designed, not built

**Status: specification only.** No code exists for this. It is recorded here because the design
decisions are the valuable part and several of them are counter-intuitive.

Exits are **not** managed by this loop. They keep the stepped-limit path, whose urgency is keyed
to the exit reason. Entry repricing and exit management are tuned separately and must never
share a cadence key — see `execution.entry_reprice_cadence_seconds`.

### Why passive entries at all

The 25 Aug fill study measured mid limits on both sides: buys filled 55%, sells 18%, and the
distribution is **bimodal, not gradual** — six of eight fills landed under a second and nothing
filled between 11s and 60s. Patience at a fixed price buys almost nothing. That is the whole
justification for stepping rather than resting, and for stepping *early*.

### Phase 1 — active window (~5 minutes)

Small steps from the mid toward the ask.

**The step cap is derived, never a fixed number of ticks or a fixed fraction of the spread.**
Step only while the trade's modelled reward-to-risk stays above the threshold that qualified
it as a candidate in the first place. Recompute at every step: both breakeven distance and
modelled P&L degrade as the entry price rises, so the ceiling is a property of the individual
contract, not of the chain.

The consequence is the point: **a high-edge contract can absorb more spread than a marginal
one.** Stop stepping when the trade stops being worth taking at that price — which is a
different price for every candidate, and the only honest definition of "too far".

### Phase 2 — pending

Rests unchased until filled or displaced. No repricing, no chasing.

### Displacement is capacity-driven, never staleness-driven

**The underlying moving away is not a reason to cancel.** On a multi-session swing thesis, the
underlying moving in the direction we predicted means the signal was *correct* and the entry
was missed. Cancelling there would systematically discard the theses that were right — the
exact inversion of what an entry manager is for.

So:

- **If cap headroom remains, keep every pending order and add the new one.** Nothing is
  displaced merely because something newer arrived.
- **Only when headroom is full** does a new signal displace an existing pending order.

### Which pending order gets cancelled

A **configurable ordered list**, evaluated in order. Default:

1. **price moved beyond a configured threshold from placement** — not staleness, but a
   quantified move that says the resting price is no longer meaningful
2. **weakest conviction** — Agent 1 confidence × Agent 2 bias strength
3. **oldest**

**`worst_current_edge` is implemented but OFF by default, deliberately.** The edge model is
what drove the DTE decision, and it has never been validated against realised outcomes — only
against its own internal consistency. Letting it silently rank live orders would give an
unvalidated model authority over real cancellations, and its errors would be invisible because
the counterfactual is never observed. It stays available and stays off until it has been
checked against realised P&L. **Turning it on is a decision that requires that evidence
first.**

### Two hard bounds, regardless of the ordering above

1. **A pending order expires after a configured number of sessions.**
2. **A pending order is cancelled if its contract falls outside the prefilter's current
   monthly/DTE band.**

The second is the load-bearing one: **a resting order must always be for something the
prefilter would still select today.** The expiry rule chooses the nearest standard monthly at
least `monthly_min_sessions` out, and that target rolls forward as sessions pass. An order left
resting across a roll would be an order for a contract the system would no longer choose —
holding it because it was valid when placed is exactly the drift these bounds exist to prevent.

### Cancellation is asynchronous, and the race is real

No replacement order is placed until every outstanding cancel is **confirmed dead** — polled to
a terminal status, not assumed from the cancel call returning.

**A cancel that returns already-filled reduces the headroom for the new batch.** That fill is a
real position; treating the cancel as successful and placing the replacement anyway would open
one more position than the caps allow. Headroom is therefore recomputed from **reconciled
broker state** after cancellation completes, never from local tracking — local counters are
exactly what the race corrupts.

### Configuration

Every threshold above is configurable and **all of them are currently unset**, under
`execution:` in `config/limits.yaml` with names in `limits.example.yaml`. Reading an unset key
raises `ConfigError` naming it, so the entry manager cannot start half-configured. They are
unset rather than defaulted because each is a policy choice this project has not yet earned the
evidence to make — the fill study constrains the cadence, and nothing yet constrains the rest.

## Exits

Deterministic exits, always armed, independent of any model:

- Hard stop: configurable, starting point −40% of premium paid
- Profit target: configurable, starting point +75% of premium paid
- **Max hold: 5 trading sessions.** Flat regardless of P&L.
- **Hard exit before expiry week.** Never hold into the final week.
- **No intraday flat rule.** Positions are held overnight by design.

Two constraints specific to the horizon:

1. **The stop can be gapped through.** A hard stop at −40% does not guarantee a −40% loss when
   the underlying gaps overnight. Sizing must assume the realised loss can exceed the stop, and
   the write-up should say so plainly.
2. **There is no broker-side stop on a spread**, and our polling loop cannot run overnight.
   Multi-session spread positions are therefore unprotected between sessions. Agent 5 and the
   deterministic exit layer must both treat this as a known, disclosed limitation rather than
   assuming coverage.

---

## Earnings exclusion

An intraday trade never touches an earnings print. A 1–5 session hold straddles one routinely.
Two separate exclusions are required, and they are not the same test.

**1. Hold-window exclusion (hard, code, pre-model).** Exclude any symbol whose next earnings
date falls within `max_hold_sessions + buffer_sessions` of the entry session. With a 5-session
max hold and a 2-session buffer, that is any symbol reporting within 7 trading sessions. Runs in
code before Agent 2 is called. No model override.

**2. Contract-span exclusion (the subtle one).** A contract whose *expiry* spans an earnings date
carries elevated implied volatility, because the market is pricing the event. Buying it means
paying an event premium for a thesis that has nothing to do with the event — and the premium is
priced into every contract in that expiry whether or not you hold through the print. So even
when the hold window is clear, prefer contracts whose expiry precedes the earnings date. Where
the chosen DTE band makes that impossible, flag it: the contract is systematically more
expensive than its realised volatility justifies, and `iv_vs_rv20` will show it.

**3. Post-earnings IV crush (the mirror image, and the one that is easy to miss).** The two
exclusions above both look *forward* to a print we might hold into. The risk does not end when
the print does. Implied volatility is bid up into an earnings date and collapses within hours
of it — frequently 30–50% of the pre-print IV on a single name. For a **long premium**
strategy that collapse is a direct loss on every open contract, and it is uncorrelated with
direction: the thesis can be right, the underlying can move the way we said, and the position
still loses because the vega component reprices. A debit vertical is partly insulated — the
short leg crushes too — which is one of the few cases where the vertical's four bid-ask
crossings may be earned.

So the buffer is **two-sided**. Entering the session *after* a print is not a clean slate; it
is the single worst moment to buy premium on that name. `earnings.post_print_buffer_sessions`
(starting point 2) excludes for that many sessions *after* the print, counted in sessions, not
days. This is a **buffer on new entries**, not an exit rule: a position already open through a
print is governed by the deterministic exits, and the crush is one of the reasons the
hold-window exclusion exists in the first place.

Three things this depends on, all enforced:

- **It needs the previous print date, not just the next one.** Once a print passes, the next
  date jumps a quarter out and every forward-looking check reads clear on the exact session IV
  is collapsing. `EarningsEntry.previous_date` carries it, read from the same symbol-scoped
  payload — which returns past quarters alongside the upcoming one, and is the reason the
  question can be asked at all.
- **An unknown previous date excludes**, and `assert_universe_resolves()` refuses to start a
  session without one while the buffer is active. Otherwise a provider that quietly stopped
  returning past quarters would disarm the buffer while every forward check kept reporting
  healthy — the same failure shape as the endpoint bug, one field over.
- **The trading calendar must reach back past the print.** `sessions_until` cannot count across
  days that were never fetched, and counting from the window edge would report a months-old
  print as days old — a false exclusion indistinguishable from a real one. `TradingCalendar.
  around()` takes `back_days` (default 7, which is forward-looking only); anything measuring
  backwards must widen it. Where the window still cannot place the print outside the buffer,
  the verdict fails closed and says to widen it.

**The longer the DTE band, the more contracts span an earnings print** — a 4-month contract on
almost any single name spans at least one. Longer-dated contracts also carry more vega, so they
absorb more of the crush.

**Data source.** Earnings dates are **not available from Alpaca's Trading API.** They come from
FMP via `src/earnings/calendar.py`. Fail closed: an unknown earnings date is treated as an
earnings date.

Two things about that provider, learned the expensive way on 2026-08-23 and enforced in code
since:

- **The endpoint must be symbol-scoped.** `/stable/earnings-calendar` accepts a `symbol`
  parameter and silently ignores it, then caps the response to a slice of the requested range.
  Filtering that client-side returns "the earliest row for this symbol that survived the cap",
  which is not the symbol's next earnings date and is *indistinguishable from one*. It read no
  date for NVDA three days before NVDA reported. Use `/stable/earnings?symbol=X`. The fetcher
  now raises if a response contains no rows for the symbol it asked about.
- **Fail-closed hides its own failure.** An unknown date excludes, so a completely dead feed
  never places a bad trade — and therefore looks exactly like a quiet week. That is why
  `assert_universe_resolves()` exists: every universe symbol must resolve to a real date or to
  an explicit `no_earnings` declaration in `config/universe.yaml`, never silently to unknown.
  Run `python -m cli.earnings_check` before any session that can place an order.

**The `no_earnings` class.** Index and sector ETFs have no print, so rule 1 would block SPY and
QQQ permanently behind a healthy-looking exclusion. They are declared in
`config/universe.yaml: no_earnings`. It is a claim about the *instrument*, never a way to skip
the check — a declared symbol that returns a real date is a contradiction and excludes loudly.
Never resolve a missing date by adding the symbol to this list.

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

`iv_assessment` must account for **where the symbol sits relative to its earnings cycle, on
both sides of the print** — this is prompt guidance, not a code constraint, because it is a
judgment and the code has no business making it. Two symbols with identical `iv_vs_rv20` are
not equally rich:

- **Approaching a print** — IV is elevated because the market is pricing a known event. That is
  not "rich", it is correctly priced, and the premium is unrecoverable for a directional
  thesis that has nothing to do with the event. Read it as `rich` for our purposes and say why.
- **Freshly past a print** — IV has crushed and screens as `cheap`. It usually is not. Realised
  volatility collapses with it, and buying long premium into the post-print lull means paying
  for movement that the calendar says will not arrive. `cheap` here should be justified against
  something other than the IV percentile alone.

The mechanical exclusion around the print is a code-side buffer and never a model call. What
Agent 2 adds is the reading of what the level *means*, which no threshold captures.

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
  "expected_hold_sessions": 3,
  "reason": "string, <=200 chars",
  "alternate_symbol": "NVDA260904C00190000"
}
```
Enforced: every returned symbol must be in the survivor set; `short_symbol` required and
non-null when `structure == debit_vertical`, and must share expiry with `primary_symbol` and be
further OTM. On any failure, fall back to the deterministic best-ratio survivor as single-leg
and log the fallback.

Structure guidance for the prompt (not enforced in code):

- **Low IV rank, strong regime confidence, DTE short enough for convergence** → debit vertical
  is viable and reduces cost basis.
- **Elevated IV rank** → single-leg long premium is expensive; a vertical's short leg offsets
  some of that, but check `iv_vs_rv20` does not indicate the whole expiry is rich.
- **Long DTE band, or a hold window short relative to DTE** → strongly favour single-leg. The
  vertical's four bid-ask crossings will not be recovered.
- **Contract spans earnings** → note it in the reason. Prefer an expiry that does not.

A vertical crosses two bid-asks on entry and two on exit. `pct_of_max_capturable_at_hold` is the
metric that decides whether those four crossings are earned: a debit spread converges toward
max value only near expiry, so over a short hold on a long-dated contract the short leg's decay
offsets the long leg's gain and the spread barely moves. A 3:1 reward-to-risk that captures 6%
of max gain over the hold is worse than a single leg.

The survivor set handed to the model remains capped at 12. Single legs rank on modelled P&L per
unit of spread cost; verticals rank on **reward-to-risk exponentially discounted by breakeven
distance in ATRs**. The model narrows; it never widens. Any structure or DTE outside the
deterministic prefilter is not selectable, by construction.

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

**4 — Chain prefilter + metrics.** Survivor set plus the computed metrics — the six single-leg
metrics, and for verticals `max_risk`, `max_gain`, `reward_to_risk` and
`pct_of_max_capturable_at_hold`.
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
