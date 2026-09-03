"""Serve the read-only dashboard.

    python -m cli.dashboard                      # localhost:8000
    python -m cli.dashboard --host 0.0.0.0 --port 8080
    python -m cli.dashboard --log-dir logs --check

``--check`` renders every view once and exits non-zero if any of them fails,
which is the thing to run before pointing a submission URL at this.

**Binding to 0.0.0.0 exposes the decision log to the network.** That is the
point for a demo URL, and it is safe *because of what the log already
guarantees*: prompts are stored as hashes and never as text, credentials are
scrubbed by the log's own redactor, and account numbers were never written to
it. What is exposed is the reasoning -- which is the artifact worth showing.
The app itself has no controls and never constructs a broker client, so the URL
cannot be used to trade regardless of who finds it.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("dashboard")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m cli.dashboard")
    p.add_argument("--host", default="127.0.0.1",
                   help="0.0.0.0 to expose it; see the note about what that shows")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--check", action="store_true",
                   help="render every view once, report, and exit")
    p.add_argument("--url", default=None,
                   help="with --check: verify a DEPLOYED dashboard over HTTP "
                        "instead of the local app")
    p.add_argument("--reload", action="store_true")
    return p.parse_args(argv)


class _RemoteClient:
    """Minimal client over a deployed dashboard, shaped like TestClient.

    The same assertions run against a URL as against the local app, so a
    deployment is verified by the checks that gate development rather than by
    a separate and weaker set. Read-only is the one that matters most remotely:
    a public URL is where an accidental control would actually be reachable.
    """

    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    @staticmethod
    def _ssl_context():
        """A context with a working CA bundle. Verification stays ON.

        This machine's Python has no default CA file
        (``ssl.get_default_verify_paths().cafile`` is None), so urllib rejects
        a perfectly valid certificate as expired. The fix is to give it a trust
        store, not to pass ``verify=False`` -- a check that skips certificate
        verification is not a check of a public HTTPS endpoint.
        """
        import ssl

        try:
            import truststore

            return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        except ImportError:
            pass
        try:
            import certifi

            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            return ssl.create_default_context()

    def _call(self, method: str, route: str):
        import urllib.error
        import urllib.request

        request = urllib.request.Request(self.base + route, method=method.upper())
        try:
            with urllib.request.urlopen(
                request, timeout=30, context=self._ssl_context()
            ) as response:
                body = response.read()
                return type("R", (), {
                    "status_code": response.status, "content": body,
                    "text": body.decode("utf-8", "replace"),
                    "json": lambda self=None, b=body: __import__("json").loads(b),
                })()
        except urllib.error.HTTPError as exc:
            body = exc.read()
            return type("R", (), {
                "status_code": exc.code, "content": body,
                "text": body.decode("utf-8", "replace"),
                "json": lambda self=None, b=body: __import__("json").loads(b or b"{}"),
            })()

    def get(self, route):
        return self._call("GET", route)

    def post(self, route):
        return self._call("POST", route)

    def put(self, route):
        return self._call("PUT", route)

    def patch(self, route):
        return self._call("PATCH", route)

    def delete(self, route):
        return self._call("DELETE", route)


def check(log_dir: Path | None, url: str | None = None) -> int:
    """Exercise every view. Locally in-process, or over HTTP against a deploy."""
    if url:
        print(f"checking deployed dashboard: {url}")
        client = _RemoteClient(url)
    else:
        from fastapi.testclient import TestClient

        from src.dashboard.app import create_app, resolve_log_dir

        directory = resolve_log_dir(log_dir)
        print(f"log dir: {directory}")
        client = TestClient(create_app(log_dir))

    failures = []
    routes = ["/", "/healthz", "/api/sessions", "/api/status",
              "/api/timeline", "/api/traces", "/api/guardrails"]
    for route in routes:
        try:
            response = client.get(route)
            ok = response.status_code == 200
            size = len(response.content)
            print(f"  [{'OK  ' if ok else 'FAIL'}] {route:<20} {response.status_code} {size:>7} bytes")
            if not ok:
                failures.append(route)
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {route:<20} {type(exc).__name__}: {exc}")
            failures.append(route)

    # And one trace end to end, since that route is parameterised.
    traces = client.get("/api/traces").json().get("traces", [])
    if traces:
        trace_id = traces[0]["trace_id"]
        response = client.get(f"/api/trace/{trace_id}")
        ok = response.status_code == 200
        print(f"  [{'OK  ' if ok else 'FAIL'}] /api/trace/{trace_id[:22]:<9} {response.status_code}")
        if not ok:
            failures.append("trace")
    else:
        print("  [--  ] /api/trace          no traces in the log yet")

    # A dashboard for an autonomous system must not be able to act.
    mutations = [m for m in ("post", "put", "patch", "delete")
                 if getattr(client, m)("/api/status").status_code not in (404, 405)]
    if mutations:
        print(f"  [FAIL] mutating verbs accepted: {mutations}")
        failures.append("read-only")
    else:
        print("  [OK  ] read-only          POST/PUT/PATCH/DELETE all rejected")

    # What a public URL must never serve. Checked against the payloads the
    # dashboard actually returns, not against the files that built it.
    body = client.get("/api/timeline").text + client.get("/api/traces").text
    leaks = [m for m in ("thresholds", "You are the", "sk-ant-", "ALPACA_", "PKTEST")
             if m in body]
    if leaks:
        print(f"  [FAIL] no IP in payloads  found: {leaks}")
        failures.append("payload-leak")
    else:
        print("  [OK  ] no IP in payloads  no thresholds, prompts or credentials")

    print()
    print(f"{'CHECK FAILED: ' + ', '.join(failures) if failures else 'CHECK OK'}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.check:
        return check(args.log_dir, args.url)

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed -- pip install 'uvicorn[standard]'", file=sys.stderr)
        return 2

    from src.dashboard.app import create_app, resolve_log_dir

    directory = resolve_log_dir(args.log_dir)
    print(f"serving {directory} at http://{args.host}:{args.port}")
    if args.host == "0.0.0.0":  # noqa: S104 -- deliberate, and explained
        print("exposed on all interfaces: read-only, no controls, no broker client")

    uvicorn.run(create_app(args.log_dir), host=args.host, port=args.port,
                log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
