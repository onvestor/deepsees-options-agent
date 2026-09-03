# Deploying the dashboard

Live: **https://deepsees-dashboard-production.up.railway.app**

A static demo of completed sessions. The logs are baked into the image at build
time; there is no volume and no live feed, and the container never reaches a
broker — `src/agents`, `src/brokers`, `src/orchestrator` and the Alpaca and
Anthropic SDKs are all absent from it.

```
python -m deploy.bundle_logs                       # sanitise logs/ -> deploy/logs/
python -m deploy.stage --out <dir outside repo>    # assemble the build context
cd <dir> && railway up --service deepsees-dashboard
python -m cli.dashboard --check --url https://...  # verify the deployment
```

## Why the staging directory must be outside the repository

The first attempt built from `deploy/_stage/` *inside* the repo and silently
inherited the repository's `.dockerignore`, whose `logs/` deny rule dropped the
sessions. The upload was 16 kB and the build failed on `COPY logs/`. Staging
outside removes the interaction entirely: what is in the directory is exactly
what ships, and `deploy/stage.py` refuses an `--out` inside the repo.

## What is stripped, and why

`deploy/bundle_logs.py` removes two fields from every record:

* **`thresholds`** — the complete tuned calibration (delta band, DTE rule,
  every metric acceptance band). `CLAUDE.md` names tuned thresholds on the
  never-publish list. The log records them so a *private* log explains itself
  without `config/`; that reasoning does not survive the log becoming public.
* **`rejections`** — per-contract failing reasons; the same calibration read
  backwards.

**Neither is rendered by any view**, so stripping them costs the demo nothing.
Verified by grep against `reader.py`, `app.py` and `index.html`, not by memory.

The bundler also refuses to run if prompt text appears in any record. Prompts
were never logged — only their sha256 — and this asserts it rather than
trusting it.

## What is deliberately public

The agents' reasoning, every skip and its reason, the guardrail events, the
orders and fills, and the account **suffix** (last four). The suffix is what
lets the dashboard prove which account a session traded rather than asserting
it by date. `/api/docs` serves the FastAPI schema page: GET routes only, no
data beyond what the API already returns.
