"""`src/signals/` must stay I/O-free. Enforced, not trusted.

CLAUDE.md makes this a structural rule: pure functions over dataframes are
what make the replay harness possible, and the replay harness is what lets
prompt iteration happen offline instead of burning live market sessions.

A comment saying "no I/O here please" survives exactly one hurried afternoon.
These tests fail the build instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SIGNALS_DIR = Path(__file__).parent.parent / "src" / "signals"
# CLAUDE.md: "src/signals/ and src/options/metrics.py are pure and must have
# direct unit tests." metrics.py is held to the same standard and checked here.
EXTRA_PURE = (Path(__file__).parent.parent / "src" / "options" / "metrics.py",)

# Anything that can reach the network, the filesystem, the clock, or a
# subprocess. `datetime` is absent deliberately -- reading the wall clock
# inside a pure function makes replay non-reproducible.
FORBIDDEN_IMPORTS = frozenset(
    {
        "alpaca", "anthropic", "requests", "httpx", "urllib", "urllib3", "http",
        "socket", "ssl", "asyncio", "aiohttp", "websocket", "websockets",
        "os", "io", "pathlib", "shutil", "tempfile", "subprocess", "sqlite3",
        "pickle", "yaml", "json", "csv", "open", "time", "datetime", "random",
    }
)

FORBIDDEN_CALLS = frozenset({"open", "print", "input", "eval", "exec", "compile"})


def signal_modules() -> list[Path]:
    modules = [p for p in SIGNALS_DIR.glob("*.py") if p.name != "__init__.py"]
    modules.extend(p for p in EXTRA_PURE if p.exists())
    return sorted(modules)


def test_there_are_signal_modules_to_check():
    assert signal_modules(), "expected modules under src/signals/"


@pytest.mark.parametrize("module", signal_modules(), ids=lambda p: p.name)
def test_no_forbidden_imports(module: Path):
    tree = ast.parse(module.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append(node.module.split(".")[0])

    offenders = sorted(set(found) & FORBIDDEN_IMPORTS)
    assert not offenders, f"{module.name} imports {offenders}; src/signals must stay I/O-free"


@pytest.mark.parametrize("module", signal_modules(), ids=lambda p: p.name)
def test_no_io_calls(module: Path):
    tree = ast.parse(module.read_text(encoding="utf-8"))
    offenders = sorted(
        {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FORBIDDEN_CALLS
        }
    )
    assert not offenders, f"{module.name} calls {offenders}; use structured logging elsewhere"


@pytest.mark.parametrize("module", signal_modules(), ids=lambda p: p.name)
def test_only_depends_on_the_numeric_stack(module: Path):
    """First-party imports must not drag the broker or agent layers in."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    internal = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src.")
    ]
    allowed = ("src.signals", "src.options.metrics")
    for name in internal:
        assert name.startswith(allowed), (
            f"{module.name} imports {name}; pure modules may only import from "
            f"{allowed} (config is passed in as a value, never read here)"
        )


def test_importing_signals_opens_no_socket(monkeypatch):
    """Belt and braces: importing the package must not touch the network."""
    import socket

    def explode(*args, **kwargs):
        raise AssertionError("src.signals opened a socket at import time")

    monkeypatch.setattr(socket, "socket", explode)
    monkeypatch.setattr(socket, "create_connection", explode)

    import importlib

    import src.signals.engine
    import src.signals.indicators

    importlib.reload(src.signals.indicators)
    importlib.reload(src.signals.engine)
