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

## Session handoff — 22 Aug 2026

State at the end of the first build session. Read this first.

### Build status

| Step | State |
| --- | --- |
| 0 — Repo hygiene | Done. `fedf4d2` |
| 1 — Broker round trip | **Code complete, UNVERIFIED.** `8249cd3`. Every read path proven; no order has ever been placed. Needs a regular session. |
| 2 — OCC, contracts, quotes, cache | Done. `7f7dfd6` |
| 3 — Signal engine | Done. `5f424ef`. Timeframe-agnostic; the swing revision changes only the bar frame and cadence, not this code. |
| 4 — Prefilter + metrics | Done. `e01d264`, extended by `06dc7fa` (vertical metrics). |
| 5 — Risk engine | Done. `ae38523` |
| 6 — Decision logger | Done. `bd94998` (built early, deliberately). |
| 7 — Agent layer | **Not started.** Blocked on the DTE decision and the earnings key. |
| 8 — Replay harness | Not started. |
| 9 — Orchestrator, CLI, dashboard | Not started. |

~1190 offline tests, 10 live (`pytest -m live`). The suite is offline by default.

### Open questions — both decided by Monday's measurement, neither by opinion

Run `python -m cli.monday_measurement --json out.json` during the session. It
sweeps three DTE buckets x three strike windows across SPY, NVDA and AMD.

**1. DTE band.** Deliberately NOT fixed. `prefilter.dte_min/max` still hold
intraday-era values (1–10 sessions) and are wrong for a 1–5 session swing hold.
The measurement decides the band; total friction per unit of delta is the
number that decides it, not theta alone.

**2. Strike window.** `prefilter.strike_window_pct` is 0.10. After the delta
band moved to 0.55–0.75 the survivor set fell to 8 on SPY and 5 on NVDA, which
is too thin for Agent 4 to make a real choice. The sweep says whether widening
recovers candidates.

**Early signal from a Friday-close smoke run — treat as a hypothesis, not a
result.** Survivor counts did not move at all across ±10/15/20% on SPY, which
suggests the binding constraint is spread and open interest rather than the
strike window. If Monday reproduces that, widening the window is not the fix.

**A third question the smoke run raised, unprompted.** At `metrics` DTE of
30–45 days the median contract cost $1,351 and at 120+ days $3,197. With
`sizing.account_risk_pct_per_trade` at 0.01 and `assume_stop_gapped` on, the
risk budget is $1,000 against a $1,351 risk-per-contract — **zero contracts
clear the caps in every bucket.** The strategy revision predicted the capital
constraint would bind; it binds harder than expected. Either
`account_risk_pct_per_trade` rises toward the revision's 5%-of-equity figure,
or the DTE band must be short enough to keep premiums affordable. Do not
resolve this by turning off `assume_stop_gapped` — that sizes larger by
pretending the overnight gap cannot happen.

### Monday queue, in order

1. **10:30 ET — greeks probe** on SPY and NVDA, before anything else. Baseline
   to beat: inside the narrowed window, SPY 78% and NVDA 99% coverage; on a
   wide chain, 58% and 47%. The result decides whether the hard reject on
   missing delta removes a residue or a third of the chain.
2. **Run the measurement harness.** Answers the DTE band, the strike window,
   and the sizing question above.
3. **Step 1 live round trip** — `python -m cli.step1_roundtrip --symbol SPY`,
   no `--dry-run`. First real order this project has placed. `ALPACA_MOCK` must
   be falsy or every write raises by design.
4. Then Step 7.

### Accounts

See the Accounts section below. `.env` currently points at the **dev** account.
The competition account is a separate login and stays **untouched until
28 Aug**. The switch is a key change only.

### Credentials

`.env` holds Alpaca (dev, paper), Anthropic, and — once added — `FMP_API_KEY`.
Validation is per consumer: broker steps do not need the Anthropic key, and
neither needs FMP. **FMP_API_KEY is set and the earnings exclusion is live**
as of 23 Aug — verified against the real provider, not mocks. See the Earnings
exclusion section for the endpoint trap that verification turned up.

### Things deliberately left undone

- **Vertical pairing.** `compute_vertical_metrics` is written and tested, but
  nothing builds candidate vertical pairs from the survivor set yet.
- **Cadence rework.** The revision moves Agent 1 and 2 to once per session
  pre-market and Agent 5 to 30 minutes. That is orchestrator work (Step 9);
  `src/signals/` needs no change.
- **Exit values.** `exits.stop_pct` and `target_pct` still hold intraday-era
  values (−35 / +60) against the revision's −40 / +75 starting points.
- **`roundtrip.*` config block.** Step 1 spike only. Delete it once the real
  order builder lands.
- **No threshold has been tuned against Friday-close data.** Spreads at the
  close are not spreads at 10:30. Everything above is measurement, not tuning.

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
