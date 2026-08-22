"""Single source of truth for configuration.

Two layers, deliberately kept apart:

* **Secrets** live in ``.env`` (or the real process environment) and are read
  into :class:`Env`. They never appear in YAML and never get logged.
* **Anything that changes trading behaviour** -- thresholds, caps, the symbol
  universe -- lives in ``config/limits.yaml`` and ``config/universe.yaml``.
  Both are gitignored. Their *key names* are documented in the committed
  ``*.example.yaml`` files; their *values* exist only on disk, locally.

Two rules this module exists to enforce:

1. **No magic numbers in source.** Every threshold is fetched by key. There is
   no ``get(key, default)`` overload, because a default at a call site is a
   magic number wearing a hat.
2. **Fail closed.** A missing file, an unparseable file, a missing key, or a
   leftover ``REPLACE_ME`` placeholder raises :class:`ConfigError` at load
   time. Nothing downstream ever sees a silently substituted value.

Usage::

    from src.config import get_config

    cfg = get_config()
    floor = cfg.limits.get_float("agents.a1.confidence_floor")
    symbols = cfg.universe.symbols
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

__all__ = [
    "AlpacaCredentials",
    "ConfigError",
    "Config",
    "Env",
    "Section",
    "Universe",
    "get_config",
    "load_config",
    "reset_config_cache",
    "REPO_ROOT",
]

# <repo>/src/config.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parent.parent

_ENV_FILE = ".env"
_LIMITS_FILE = "limits.yaml"
_UNIVERSE_FILE = "universe.yaml"

# Sentinels used throughout the committed example files. Reaching load time
# with one still in place means someone copied the example and never filled it
# in -- a hard stop, not a warning.
_PLACEHOLDERS = ("REPLACE_ME", "PLACEHOLDER", "CHANGEME", "TODO")


class ConfigError(RuntimeError):
    """Configuration is missing, malformed, or still holding a placeholder."""


# ---------------------------------------------------------------------------
# .env
# ---------------------------------------------------------------------------


def _parse_env_file(path: Path) -> dict[str, str]:
    """Minimal ``KEY=VALUE`` parser. No interpolation, no ``export`` magic.

    Deliberately not python-dotenv: the format we actually use is one line per
    key, and a 20-line parser is easier to audit than a dependency that can
    resolve shell expressions inside a secrets file.
    """
    values: dict[str, str] = {}
    if not path.is_file():
        return values

    lines = path.read_text(encoding="utf-8").splitlines()
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            raise ConfigError(f"{path}:{lineno}: expected KEY=VALUE, got {raw!r}")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


@dataclass(frozen=True)
class AlpacaCredentials:
    """Everything needed to talk to Alpaca, proven present."""

    api_key: str
    secret_key: str
    base_url: str
    data_url: str | None

    @property
    def is_paper(self) -> bool:
        return "paper-api" in self.base_url

    def __repr__(self) -> str:
        return f"AlpacaCredentials(base_url={self.base_url!r}, secrets=<redacted>)"


_TRUTHY = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class Env:
    """Credentials and endpoints. Never rendered into logs or the decision log.

    Validation is **per consumer**, not eager. Loading config must not demand
    an Anthropic key to run the broker round trip, nor Alpaca keys to run a
    nightly review over an existing decision log. Each subsystem calls the
    ``require_*`` accessor for what it actually needs, and the error names only
    the variables that subsystem is missing.

    Feeds are deliberately absent: ``broker.data_feed_options`` and
    ``broker.data_feed_equities`` live in limits.yaml, which is the single
    source for anything that changes behaviour. ``.env`` holds credentials and
    endpoints only.
    """

    alpaca_api_key: str | None
    alpaca_secret_key: str | None
    alpaca_base_url: str | None
    alpaca_data_url: str | None
    anthropic_api_key: str | None
    fmp_api_key: str | None
    mock: bool

    @property
    def is_paper(self) -> bool:
        """True only when demonstrably pointed at paper. Unset fails closed."""
        return bool(self.alpaca_base_url) and "paper-api" in self.alpaca_base_url

    def require_alpaca(self) -> AlpacaCredentials:
        """Credentials for any broker or market-data call."""
        missing = [
            name
            for name, value in (
                ("ALPACA_API_KEY", self.alpaca_api_key),
                ("ALPACA_SECRET_KEY", self.alpaca_secret_key),
                ("ALPACA_BASE_URL", self.alpaca_base_url),
            )
            if not value
        ]
        if missing:
            raise ConfigError(
                "broker access needs " + ", ".join(missing) + " -- set them in .env "
                "(see .env.example)"
            )
        return AlpacaCredentials(
            api_key=self.alpaca_api_key,  # type: ignore[arg-type]
            secret_key=self.alpaca_secret_key,  # type: ignore[arg-type]
            base_url=self.alpaca_base_url,  # type: ignore[arg-type]
            data_url=self.alpaca_data_url,
        )

    def require_fmp(self) -> str:
        """Key for the earnings calendar. Demanded only by the exclusion."""
        if not self.fmp_api_key:
            raise ConfigError(
                "the earnings exclusion needs FMP_API_KEY -- set it in .env "
                "(see .env.example). Running without it would trade through "
                "earnings prints."
            )
        return self.fmp_api_key

    def require_anthropic(self) -> str:
        """Key for a model call. Demanded only when an agent is invoked."""
        if not self.anthropic_api_key:
            raise ConfigError(
                "agent calls need ANTHROPIC_API_KEY -- set it in .env (see .env.example). "
                "Broker-only workflows do not require it."
            )
        return self.anthropic_api_key

    def __repr__(self) -> str:
        return (
            f"Env(alpaca_base_url={self.alpaca_base_url!r}, "
            f"alpaca_data_url={self.alpaca_data_url!r}, mock={self.mock}, "
            f"alpaca_keys={'set' if self.alpaca_api_key else 'unset'}, "
            f"anthropic_key={'set' if self.anthropic_api_key else 'unset'}, "
            f"fmp_key={'set' if self.fmp_api_key else 'unset'}, "
            "secrets=<redacted>)"
        )


def _load_env(repo_root: Path) -> Env:
    """Read ``.env`` and the environment. Validates nothing -- see ``require_*``.

    Real process environment wins over ``.env`` so hosted overrides work.
    """
    from_file = _parse_env_file(repo_root / _ENV_FILE)

    def read(key: str) -> str | None:
        value = (os.environ.get(key) or from_file.get(key) or "").strip()
        return value or None

    base_url = read("ALPACA_BASE_URL")
    data_url = read("ALPACA_DATA_URL")
    mock = (read("ALPACA_MOCK") or "").lower() in _TRUTHY

    return Env(
        alpaca_api_key=read("ALPACA_API_KEY"),
        alpaca_secret_key=read("ALPACA_SECRET_KEY"),
        alpaca_base_url=base_url.rstrip("/") if base_url else None,
        alpaca_data_url=data_url.rstrip("/") if data_url else None,
        anthropic_api_key=read("ANTHROPIC_API_KEY"),
        fmp_api_key=read("FMP_API_KEY"),
        mock=mock,
    )


# ---------------------------------------------------------------------------
# YAML sections
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        example = path.name.replace(".yaml", ".example.yaml")
        raise ConfigError(f"{path} not found -- copy {example} to {path.name} and fill it in")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if data is None:
        raise ConfigError(f"{path} is empty")
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be a mapping at the top level, got {type(data).__name__}")
    return data


def _assert_no_placeholders(node: Any, path: Path, trail: str = "") -> None:
    """Reject leftover example sentinels and nulls anywhere in the tree.

    A ``null`` in a real config file is almost always a key copied from the
    example and never filled in. Failing here is what stops it becoming a
    ``None`` comparison somewhere inside the risk layer.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            _assert_no_placeholders(value, path, f"{trail}.{key}" if trail else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _assert_no_placeholders(value, path, f"{trail}[{index}]")
    elif node is None:
        raise ConfigError(f"{path}: {trail!r} is null -- fill in a real value")
    elif isinstance(node, str):
        upper = node.strip().upper()
        if any(upper.startswith(marker) for marker in _PLACEHOLDERS):
            raise ConfigError(f"{path}: {trail!r} still holds the placeholder {node!r}")


class Section:
    """Read-only view over one YAML file, addressed by dotted key.

    There is no ``default`` parameter anywhere in this API, by design: a caller
    that wants a fallback has to add the key to the config file, so the value
    stays reviewable in one place instead of scattered through source as a
    literal.
    """

    __slots__ = ("_data", "_source")

    def __init__(self, data: Mapping[str, Any], source: Path) -> None:
        self._data = data
        self._source = source

    @property
    def source(self) -> Path:
        return self._source

    def get(self, dotted_key: str) -> Any:
        """Fetch a value. Raises :class:`ConfigError` if the key is absent."""
        node: Any = self._data
        walked: list[str] = []
        for part in dotted_key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                reached = ".".join(walked) or "<root>"
                raise ConfigError(
                    f"{self._source}: missing key {dotted_key!r} (resolved as far as {reached!r})"
                )
            node = node[part]
            walked.append(part)
        return node

    def get_float(self, dotted_key: str) -> float:
        value = self.get(dotted_key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{self._source}: {dotted_key!r} must be a number, got {value!r}")
        return float(value)

    def get_int(self, dotted_key: str) -> int:
        value = self.get(dotted_key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{self._source}: {dotted_key!r} must be an integer, got {value!r}")
        return value

    def get_bool(self, dotted_key: str) -> bool:
        value = self.get(dotted_key)
        if not isinstance(value, bool):
            raise ConfigError(f"{self._source}: {dotted_key!r} must be a boolean, got {value!r}")
        return value

    def get_str(self, dotted_key: str) -> str:
        value = self.get(dotted_key)
        if not isinstance(value, str):
            raise ConfigError(f"{self._source}: {dotted_key!r} must be a string, got {value!r}")
        return value

    def get_list(self, dotted_key: str) -> list[Any]:
        value = self.get(dotted_key)
        if not isinstance(value, list):
            raise ConfigError(f"{self._source}: {dotted_key!r} must be a list, got {value!r}")
        return list(value)

    def get_int_set(self, dotted_key: str) -> frozenset[int]:
        """For the allowed-value sets Agent 1's signal profile is clamped to."""
        values = self.get_list(dotted_key)
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(
                    f"{self._source}: {dotted_key!r} must contain integers, got {value!r}"
                )
        return frozenset(values)

    def get_str_set(self, dotted_key: str) -> frozenset[str]:
        values = self.get_list(dotted_key)
        for value in values:
            if not isinstance(value, str):
                raise ConfigError(
                    f"{self._source}: {dotted_key!r} must contain strings, got {value!r}"
                )
        return frozenset(values)

    def has(self, dotted_key: str) -> bool:
        try:
            self.get(dotted_key)
        except ConfigError:
            return False
        return True

    def as_dict(self) -> dict[str, Any]:
        """Deep copy, so no caller can mutate loaded config in place."""
        return copy.deepcopy(dict(self._data))


class Universe(Section):
    """``config/universe.yaml``: the symbol list plus per-symbol overrides."""

    @property
    def symbols(self) -> tuple[str, ...]:
        raw = self.get_list("symbols")
        symbols: list[str] = []
        for entry in raw:
            if not isinstance(entry, str) or not entry.strip():
                raise ConfigError(f"{self.source}: 'symbols' must contain non-empty strings")
            symbols.append(entry.strip().upper())
        if not symbols:
            raise ConfigError(f"{self.source}: 'symbols' is empty")
        duplicates = sorted({s for s in symbols if symbols.count(s) > 1})
        if duplicates:
            raise ConfigError(f"{self.source}: duplicate symbols {duplicates}")
        return tuple(symbols)

    def override(self, symbol: str, dotted_key: str) -> Any:
        """Per-symbol value, falling back to the universe-wide default.

        Both lookups go through :meth:`Section.get`, so an unknown key is still
        an error rather than a ``None``.
        """
        scoped = f"overrides.{symbol.strip().upper()}.{dotted_key}"
        if self.has(scoped):
            return self.get(scoped)
        return self.get(f"defaults.{dotted_key}")

    def __iter__(self) -> Iterator[str]:
        return iter(self.symbols)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    repo_root: Path
    config_dir: Path
    log_dir: Path
    prompt_dir: Path
    env: Env
    limits: Section
    universe: Universe

    def prompt_path(self, name: str) -> Path:
        """Path to a prompt file. Never committed; resolved at runtime only.

        A fresh clone has no ``prompts/`` -- that is the IP boundary, not a
        bug. The error therefore names the missing file and says who supplies
        it, rather than surfacing as a bare FileNotFoundError.
        """
        path = self.prompt_dir / name
        if not path.is_file():
            raise ConfigError(
                f"prompt {name!r} not found at {path} -- prompts/ is operator-supplied "
                "and is not part of the repository (see README)"
            )
        return path

    def ensure_cache_dir(self) -> Path:
        """Create ``cache/`` on first use. Provider data, gitignored."""
        path = _resolve_dir("DEEPSEES_CACHE_DIR", self.repo_root / "cache")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_log_dir(self) -> Path:
        """Create ``logs/`` on first write. It is gitignored, so a fresh clone
        never has one."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self.log_dir


def _resolve_dir(env_var: str, default: Path) -> Path:
    raw = os.environ.get(env_var, "").strip()
    return Path(raw).expanduser().resolve() if raw else default


def load_config(repo_root: Path | None = None) -> Config:
    """Load and validate everything. Raises :class:`ConfigError` on any problem.

    Not cached -- tests and the replay harness use this against a temporary
    tree. Application code should call :func:`get_config`.
    """
    root = (repo_root or REPO_ROOT).resolve()

    config_dir = _resolve_dir("DEEPSEES_CONFIG_DIR", root / "config")
    log_dir = _resolve_dir("DEEPSEES_LOG_DIR", root / "logs")
    prompt_dir = _resolve_dir("DEEPSEES_PROMPT_DIR", root / "prompts")

    limits_path = config_dir / _LIMITS_FILE
    universe_path = config_dir / _UNIVERSE_FILE

    limits_data = _load_yaml(limits_path)
    universe_data = _load_yaml(universe_path)
    _assert_no_placeholders(limits_data, limits_path)
    _assert_no_placeholders(universe_data, universe_path)

    config = Config(
        repo_root=root,
        config_dir=config_dir,
        log_dir=log_dir,
        prompt_dir=prompt_dir,
        env=_load_env(root),
        limits=Section(limits_data, limits_path),
        universe=Universe(universe_data, universe_path),
    )
    # Touch the symbol list once at load, so a malformed universe fails here
    # rather than mid-session.
    _ = config.universe.symbols
    return config


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide config. Cached: the files are read once per process."""
    return load_config()


def reset_config_cache() -> None:
    """Drop the cached config. For tests and the replay harness only."""
    get_config.cache_clear()
