"""Read-only dashboard over the decision log. FastAPI, one page, no build step.

**There are no controls, and that is a design position rather than a shortcut.**
A "close position" button on a dashboard for an autonomous system is a
contradiction: either the system decides, or a human does. Every route here is
a GET, nothing mutates, and the app never constructs a broker client at all --
so there is no code path by which a click could reach an order, even by
accident.

**No broker calls.** Every figure is derived from ``decision_log.jsonl``. That
is what makes the dashboard an audit trail rather than a second, unverifiable
view of the account: if a number is on the screen, it can be traced to a line
in the log. The "live status" panel is therefore the last *recorded* state, and
it shows its own staleness rather than pretending to be current.

**The log is re-read per request.** No caching, so a session appending to the
file shows up on the next refresh. The files are small (a session is a few
thousand lines) and a stale dashboard during a live session would be worse than
a slow one.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from src.dashboard.reader import Log, discover

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"


def resolve_log_dir(explicit: Path | None = None) -> Path:
    """Where the decision logs live.

    Falls back to ``config.log_dir`` so the dashboard follows the same
    ``DEEPSEES_LOG_DIR`` the session writes to. Config is read for the *path*
    only -- no threshold on this page comes from config, because then the page
    would be showing something the log cannot prove.
    """
    if explicit:
        return Path(explicit)
    try:
        from src.config import load_config

        return load_config().log_dir
    except Exception:  # noqa: BLE001 -- a dashboard must start without config
        return Path("logs")


def create_app(log_dir: Path | None = None) -> FastAPI:
    directory = resolve_log_dir(log_dir)
    app = FastAPI(
        title="DeepSees Options Agent",
        description="Read-only decision log viewer. No controls, no broker calls.",
        version="1.0",
        docs_url="/api/docs",
    )

    def read(session: str | None) -> Log:
        return Log.load(discover(directory, session))

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        page = STATIC / "index.html"
        if not page.is_file():
            raise HTTPException(500, f"dashboard page missing at {page}")
        return page.read_text(encoding="utf-8")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        files = discover(directory)
        return {
            "ok": True,
            "log_dir": str(directory),
            "log_files": [f.name for f in files],
        }

    @app.get("/api/sessions")
    def sessions() -> dict[str, Any]:
        data = read(None)
        return {"sessions": data.sessions, "sources": data.sources}

    @app.get("/api/timeline")
    def timeline(
        session: str | None = Query(None),
        kind: str | None = Query(None),
        symbol: str | None = Query(None),
        guardrails_only: bool = Query(False),
    ) -> dict[str, Any]:
        rows = read(session).timeline(session)
        if kind:
            rows = [r for r in rows if r["kind"] == kind]
        if symbol:
            rows = [r for r in rows if r["symbol"] == symbol.upper()]
        if guardrails_only:
            rows = [r for r in rows if r["guardrail"]]
        return {"session": session, "count": len(rows), "rows": rows}

    @app.get("/api/traces")
    def traces(session: str | None = Query(None)) -> dict[str, Any]:
        found = read(session).traces(session)
        # The chain is heavy; the list view only needs the headline.
        return {
            "session": session,
            "count": len(found),
            "traces": [{k: v for k, v in t.items() if k != "chain"} for t in found],
        }

    @app.get("/api/trace/{trace_id}")
    def trace(trace_id: str, session: str | None = Query(None)) -> dict[str, Any]:
        found = read(session).trace(trace_id, session)
        if found is None:
            raise HTTPException(404, f"no trace {trace_id!r}")
        return found

    @app.get("/api/guardrails")
    def guardrails(session: str | None = Query(None)) -> dict[str, Any]:
        return read(session).guardrails(session)

    @app.get("/api/status")
    def status(session: str | None = Query(None)) -> dict[str, Any]:
        return read(session).status(session)

    @app.exception_handler(404)
    def not_found(_request, exc):  # noqa: ANN001
        return JSONResponse({"error": str(getattr(exc, "detail", "not found"))}, 404)

    return app


app = create_app()
