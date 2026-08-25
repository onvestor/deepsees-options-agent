"""Load and render operator-supplied prompt templates by name.

Prompt text is IP and lives in the gitignored ``prompts/``. Modules reference a
prompt by **name** only; nothing in this repository ever contains the text, and
a fresh clone fails with a message naming the missing file rather than a stack
trace. :meth:`Config.prompt_path` already enforces that half.

**Substitution uses ``$field``, not ``{field}``.** Prompt templates contain
JSON examples -- that is most of what an agent contract prompt *is* -- and
``str.format`` would choke on every brace in them. ``string.Template`` leaves
braces alone.

**Missing fields raise.** ``substitute`` is used rather than
``safe_substitute`` because a template referencing a field the caller did not
supply should fail loudly, not send the model a prompt with a literal
``$atr`` in it and act on whatever comes back.
"""
from __future__ import annotations

from string import Template
from typing import Any

from src.config import ConfigError


def load_template(config: Any, name: str) -> str:
    """Read a prompt template by name. Raises ConfigError if absent."""
    return config.prompt_path(name).read_text(encoding="utf-8")


def render(template: str, fields: dict[str, Any]) -> str:
    """Substitute ``$field`` placeholders. Every referenced field is required."""
    try:
        return Template(template).substitute(fields)
    except KeyError as exc:
        raise ConfigError(
            f"prompt template references {exc.args[0]!r}, which the caller did not "
            f"supply. Available: {sorted(fields)}"
        ) from None
    except ValueError as exc:
        raise ConfigError(f"prompt template is malformed: {exc}") from None


def load_and_render(config: Any, name: str, fields: dict[str, Any]) -> str:
    return render(load_template(config, name), fields)
