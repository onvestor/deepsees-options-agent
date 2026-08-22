"""OCC option symbol build and parse.

Alpaca returns **unpadded** roots: ``SPY260824C00500000``, not the OSI form
``SPY   260824C00500000`` with the root space-padded to six characters.
Verified against live contracts -- 200/200 round-trip exactly, root lengths 3
and 4, no space padding anywhere.

The parse is a fixed-width split, not a regex. Everything after the root is
exactly 15 characters:

    YYMMDD (6) + C|P (1) + strike x 1000, zero-padded (8)  =  15

so ``root = symbol[:-15]`` is unambiguous even when the root itself contains
digits -- adjusted contracts (``AAPL1``) and weekly roots (``SPXW``) both fall
out correctly. A non-greedy regex over the root gets this right by accident on
plain roots and wrong on the interesting ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

__all__ = ["OccSymbol", "OccError", "TAIL_LENGTH", "build", "parse"]

# YYMMDD + C/P + 8 strike digits.
TAIL_LENGTH = 15
_STRIKE_SCALE = 1000
_STRIKE_DIGITS = 8

OptionType = Literal["call", "put"]


class OccError(ValueError):
    """A symbol is not a well-formed OCC option symbol."""


@dataclass(frozen=True)
class OccSymbol:
    root: str
    expiry: date
    option_type: OptionType
    strike: float

    def __str__(self) -> str:
        return build(self.root, self.expiry, self.option_type, self.strike)


def build(root: str, expiry: date, option_type: OptionType, strike: float) -> str:
    """Construct an unpadded OCC symbol.

    Strike is scaled by 1000 and rounded, not truncated: 0.1 + 0.2 style float
    error would otherwise turn a $17.50 strike into ``00017499``.
    """
    root = root.strip().upper()
    if not root or not root.isalnum():
        raise OccError(f"root must be non-empty alphanumeric, got {root!r}")
    if option_type not in ("call", "put"):
        raise OccError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if strike <= 0:
        raise OccError(f"strike must be positive, got {strike!r}")

    scaled = int(round(strike * _STRIKE_SCALE))
    if len(str(scaled)) > _STRIKE_DIGITS:
        raise OccError(f"strike {strike} does not fit in {_STRIKE_DIGITS} digits")

    return f"{root}{expiry:%y%m%d}{'C' if option_type == 'call' else 'P'}{scaled:0{_STRIKE_DIGITS}d}"


def parse(symbol: str) -> OccSymbol:
    """Split an OCC symbol. Total, deterministic, and never guesses.

    Anything malformed raises rather than returning a partially-populated
    result -- a wrong strike parsed silently would size a position against the
    wrong contract.
    """
    if not isinstance(symbol, str):
        raise OccError(f"expected a string, got {type(symbol).__name__}")
    text = symbol.strip().upper()
    if len(text) <= TAIL_LENGTH:
        raise OccError(f"{symbol!r} is too short to be an OCC symbol")

    root, tail = text[:-TAIL_LENGTH], text[-TAIL_LENGTH:]
    if not root.isalnum():
        raise OccError(f"{symbol!r} has a non-alphanumeric root {root!r}")

    yymmdd, cp, strike_digits = tail[:6], tail[6], tail[7:]
    if not yymmdd.isdigit() or not strike_digits.isdigit():
        raise OccError(f"{symbol!r} has a malformed date or strike field")
    if cp not in ("C", "P"):
        raise OccError(f"{symbol!r} has option type {cp!r}, expected 'C' or 'P'")

    try:
        expiry = date(2000 + int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6]))
    except ValueError as exc:
        raise OccError(f"{symbol!r} has an invalid expiry date: {exc}") from exc

    return OccSymbol(
        root=root,
        expiry=expiry,
        option_type="call" if cp == "C" else "put",
        strike=int(strike_digits) / _STRIKE_SCALE,
    )
