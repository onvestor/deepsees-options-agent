# deepsees-options-agent

Autonomous multi-agent options trading system for the Alpaca AI Trading Agents Hackathon.
Trades intraday debit structures on a curated liquid universe against Alpaca paper trading.

> **LLMs make judgments. Code makes calculations and enforces limits.**
>
> Agents emit structured intent — never a price, never a quantity, never an order. A
> validation layer sits between every agent and the broker. See [CLAUDE.md](CLAUDE.md) for the
> full architecture, the three safety invariants, and the six agent contracts.

## This repository is deliberately incomplete

Three things are **operator-supplied** and are not in the repository. This is an intentional
boundary, not a missing build step: the tuned thresholds, the symbol universe, and the agent
prompt text are the parts of the system that carry the edge.

| Path | Contents | How to create it |
| --- | --- | --- |
| `.env` | Alpaca + Anthropic credentials | `cp .env.example .env`, then fill in |
| `config/limits.yaml` | Thresholds, caps, kill switches | `cp config/limits.example.yaml config/limits.yaml`, then fill in |
| `config/universe.yaml` | Tradeable symbols, per-symbol overrides | `cp config/universe.example.yaml config/universe.yaml`, then fill in |
| `prompts/*.txt` | Runtime prompt text for agents 1–6 | Author locally; never committed |

The committed `*.example.yaml` files document every **key name** the code reads, with the type
and unit of each value in a comment. They carry placeholder values only. `src/config.py`
rejects both `null` and any leftover `REPLACE_ME` at load time, so an example file cannot be
used as a working config by accident — and a missing file fails with a message naming the file
rather than a stack trace.

`logs/` is created on first write.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
cp config/limits.example.yaml config/limits.yaml
cp config/universe.example.yaml config/universe.yaml
# fill in all three, then:
python -c "from src.config import get_config; print(get_config().universe.symbols)"
```

Paper trading only. `config/limits.yaml` carries `broker.require_paper_base_url`, and
`Env.is_paper` guards against being pointed at a live account.

## Layout

```
src/config.py        single source of truth: config/*.yaml + .env
src/brokers/alpaca/  auth, contract discovery, snapshot quotes, orders, positions
src/signals/         PURE FUNCTIONS ONLY — no I/O, no network
src/options/         OCC symbols, deterministic chain prefilter, computed metrics
src/agents/          schemas, validator, runner, agents 1–6
src/risk/            base sizing, hard caps, kill switches
src/orchestrator/    market-hours state machine, per-agent cadence
src/decisionlog/     append-only JSONL — the project's most valuable artifact
cli/                 cron entry points
replay/              offline bar replay harness
tests/fixtures/      recorded Alpaca responses
```

## Testing

```bash
pytest
```

Parsers are tested against recorded Alpaca responses, not hand-written JSON. Agent tests use
deliberately malformed model responses — the fail-closed paths matter more than the happy
path. The suite never places live paper orders.

## License

See [LICENSE](LICENSE).
