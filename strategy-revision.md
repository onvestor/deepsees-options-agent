# CLAUDE.md — strategy revision

Replace the corresponding sections in `CLAUDE.md`. Everything not mentioned here is unchanged.
All numeric values live in `config/limits.yaml`; the bands below are starting points to be set
by measurement, not constants to hardcode.

---

## Strategy (replaces the opening description)

Autonomous **swing** options trading. Directional theses on a curated liquid universe,
expressed as **debit structures** rather than shares — long calls, long puts, and debit
verticals. The option is leverage on a stock-like directional view, not a volatility bet.

**Hold horizon: 1–5 trading sessions.** This is a positional system. There is no intraday
flat rule.

Design consequences that follow from the horizon, and that must not drift:

- **Theta matters, but less than friction.** Over a 3-session hold on a 45–90 DTE contract,
  theta is a modest cost. Round-trip spread cost on an illiquid contract frequently exceeds
  it. Optimise total friction, not theta alone.
- **Vega is now a live risk.** Vega scales with time to expiry. A long-dated contract can lose
  more to an IV decline than it gains from a correct directional call. This is uncompensated
  risk for a directional strategy.
- **Overnight gap risk is real and unhedgeable by polling.** Positions are held through
  sessions we cannot poll. Sizing must assume the stop can be gapped through.

---

## DTE — measured, not assumed

`dte_band_days` is configurable. **Do not fix it before the Monday measurement.**

Run the comparison on SPY, NVDA and AMD across three buckets — 30–45, 60–90, 120+ days —
and report, for a modelled 3-session hold:

| Metric | Why |
|---|---|
| `theta_pct_per_day` × 3 sessions | the cost longer DTE is meant to avoid |
| `spread_cost_vs_expected_move` round trip | the cost longer DTE adds |
| total friction per unit of delta | the number that actually decides it |
| vega as % of premium | the risk longer DTE introduces |
| premium per contract | see the capital constraint below |
| contracts clearing the liquidity filters | long-dated contracts often fail OI/volume |

**The capital constraint is likely to be binding.** At a 5%-of-equity cap on a $100k account,
one position may not exceed $5,000. A 0.70-delta 4-month call on a $700+ underlying can cost
$5,500–6,500 — a single contract that cannot be bought at all, and no sizing granularity even
where it fits. Report how many contracts clear the caps in each bucket; a bucket where the
answer is "one or zero" is not viable regardless of its theta profile.

Expected outcome is that the middle bucket wins, but the measurement decides.

**Hard constraint regardless of the band chosen:** expiry must clear the maximum hold window
by a wide margin. Never manage a position in its final week — theta acceleration and widening
spreads both bite there.

---

## Delta bands

| Structure | Leg | Band |
|---|---|---|
| Single leg | — | 0.55–0.75 |
| Debit vertical | Long | 0.55–0.70 |
| Debit vertical | Short | 0.25–0.35, same expiry, further OTM |

Rationale for the prompt, not enforced in code: slightly ITM gives 55–75% participation in the
underlying move with defined risk and modest theta. Past ~0.80 delta, you are paying for
intrinsic value and crossing wider spreads for a diminishing gain over simply holding stock.

**Verticals need a viable hold window.** A debit spread converges toward maximum value only
near expiry. On a long-dated contract over a 3-session hold, a vertical barely moves — the
short leg's decay offsets the long leg's gain. Verticals are only worth their four bid-ask
crossings when the chosen DTE is short enough for convergence to contribute. If the Monday
measurement lands on a long DTE band, Agent 4's structure guidance should favour single-leg
heavily.

---

## Signal cadence (replaces the 5-minute intraday design)

A 9 EMA cross on 5-minute bars is an intraday trigger and is no longer appropriate.

- **Signal evaluation:** daily bars, with hourly as a secondary confirmation timeframe.
- **Agent 1 (regime):** once per session pre-market, not every 30 minutes. Regime for a
  multi-session hold is a daily judgment. The 30-minute profile lock is replaced by a
  full-session lock.
- **Agent 2 (context):** once per session pre-market, unchanged in shape.
- **Agent 5 (exit):** every 30 minutes per open position, plus immediately on a ±20% premium
  move. Not every 5 minutes.

This cuts the call budget by roughly an order of magnitude — from ~250–400 per session to
~30–60. Latency tolerances can relax accordingly.

`src/signals/` is unaffected: the indicators are timeframe-agnostic pure functions. Only the
bar frame passed in and the cadence of evaluation change.

---

## Exits (replaces the intraday exit block)

Deterministic exits, always armed, independent of any model:

- Hard stop: configurable, starting point −40% of premium paid
- Profit target: configurable, starting point +75% of premium paid
- **Max hold: 5 trading sessions.** Flat regardless of P&L.
- **Hard exit before expiry week.** Never hold into the final week.
- **No intraday flat rule.** Positions are held overnight by design.

Two constraints specific to the horizon:

1. **The stop can be gapped through.** A hard stop at −40% does not guarantee a −40% loss when
   the underlying gaps overnight. Sizing must be set assuming the realised loss can exceed the
   stop, and the write-up should say so plainly.
2. **There is no broker-side stop on a spread.** Alpaca's `stop` and `stop_limit` order types
   are single-leg only. Every vertical exit is managed by our own polling loop sending a
   closing `mleg` order — and that loop cannot run overnight. Multi-session spread positions
   are therefore unprotected between sessions. Agent 5 and the deterministic exit layer must
   both treat this as a known, disclosed limitation rather than assuming coverage.

Agent 5's permitted actions are unchanged: `hold | tighten_stop | scale_out_half | exit_now`.
The monotone invariant holds — it may only tighten, never widen.

---

## Earnings exclusion (expanded — this is now critical)

An intraday trade never touches an earnings print. A 1–5 session hold straddles one routinely.
Two separate exclusions are required, and they are not the same test.

**1. Hold-window exclusion (hard, code, pre-model).**

Exclude any symbol whose next earnings date falls within `max_hold_sessions + buffer_sessions`
of the entry session. With a 5-session max hold and a 2-session buffer, that is any symbol
reporting within 7 trading sessions. This runs in code before Agent 2 is called, and there is
no model override.

**2. Contract-span exclusion (the subtle one).**

A contract whose *expiry* spans an earnings date carries elevated implied volatility, because
the market is pricing the event. Buying it means paying an event premium for a thesis that has
nothing to do with the event — and the premium is priced into every contract in that expiry
whether or not you hold through the print.

So even when the hold window is clear, prefer contracts whose expiry precedes the earnings
date. Where the chosen DTE band makes that impossible, flag it: the contract is systematically
more expensive than its realised-volatility justifies, and `iv_vs_rv20` will show it.

This interacts directly with the DTE decision. **The longer the DTE band, the more contracts
span an earnings print** — a 4-month contract on almost any single name spans at least one.
Include the fraction of contracts spanning earnings in each bucket of the Monday measurement.

**Data source.** Earnings dates are not available from Alpaca's Trading API. Note the gap
explicitly and decide the source before Step 7 — the exclusion is worthless if the calendar is
stale or missing. Fail closed: an unknown earnings date is treated as an earnings date.

---

## Agent 4 — revised structure guidance

Unchanged in schema. The prompt guidance becomes:

- **Low IV rank, strong regime confidence, DTE short enough for convergence** → debit vertical
  is viable and reduces cost basis.
- **Elevated IV rank** → single-leg long premium is expensive; a vertical's short leg offsets
  some of that, but check that `iv_vs_rv20` does not indicate the whole expiry is rich.
- **Long DTE band, or a hold window short relative to DTE** → strongly favour single-leg. The
  vertical's four bid-ask crossings will not be recovered.
- **Contract spans earnings** → note it in the reason. Prefer an expiry that does not.

The survivor set handed to the model remains capped at 12, ranked by modelled P&L to spread
cost. The model narrows; it never widens. Any structure or DTE outside the deterministic
prefilter is not selectable, by construction.

---

## What this changes about the trading week

Five sessions with a 1–5 session hold means roughly **2–4 completed round trips**, not twenty.

That is a smaller P&L sample and more variance — but the P&L criterion was never going to be
signal over five sessions regardless. The upside is that each trade is fully explicable on
camera, with a complete decision trace from regime through context, structure, sizing and exit.
Optimise the explanation, not the trade count.
