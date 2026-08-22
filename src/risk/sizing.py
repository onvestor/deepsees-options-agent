"""Base position sizing, computed in code before any model sees it.

The order is fixed and the direction of travel is one-way:

    risk budget -> base contracts -> model multiplier (shrink only) -> caps

Agent 3 never sees a contract count until after the code has computed one, and
its only lever is a scalar clamped to ``[0.0, 1.0]``. There is no code path
here by which a model increases exposure -- the multiplier is clamped on the
way in and the result is asserted against the base on the way out.

**Gap risk sets the risk-per-contract.** This is a swing system holding
positions across sessions we cannot poll. A hard stop at -40% does not
guarantee a -40% loss when the underlying gaps overnight, so by default sizing
assumes the whole premium can be lost regardless of where the stop sits.
``sizing.assume_stop_gapped`` can turn that off, but the honest default for a
multi-session hold is on, and the write-up should say so plainly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.risk.caps import CapVerdict, apply_caps

__all__ = [
    "AccountState",
    "SizingLimits",
    "SizingResult",
    "compute_size",
]


@dataclass(frozen=True)
class AccountState:
    """Live account and book state at the moment of sizing.

    ``options_buying_power`` is read fresh from the broker, never assumed from
    equity. On a margined account the two differ by the whole margin loan.
    """

    equity: float
    options_buying_power: float
    open_premium: float = 0.0
    open_positions: int = 0
    positions_in_symbol: int = 0
    entries_this_session: int = 0
    entries_this_symbol_this_session: int = 0

    @classmethod
    def from_account(cls, account: Any, **book: Any) -> "AccountState":
        """Build from an Alpaca account object via the guarded capital read."""
        from src.brokers.alpaca.client import sizing_capital

        return cls(
            equity=float(account.equity),
            options_buying_power=sizing_capital(account),
            **book,
        )


@dataclass(frozen=True)
class SizingLimits:
    """Every threshold sizing consults, resolved from config once."""

    account_risk_pct_per_trade: float
    max_contracts_per_trade: int
    max_premium_per_trade: float
    max_premium_pct_of_equity: float
    min_contracts: int
    assume_stop_gapped: bool
    stop_pct: float
    max_concurrent_positions: int
    max_positions_per_symbol: int
    max_open_premium: float
    max_open_premium_pct_of_equity: float
    max_entries_per_session: int
    max_entries_per_symbol_per_session: int
    size_multiplier_min: float = 0.0
    size_multiplier_max: float = 1.0

    @classmethod
    def from_limits(cls, limits: Any) -> "SizingLimits":
        return cls(
            account_risk_pct_per_trade=limits.get_float("sizing.account_risk_pct_per_trade"),
            max_contracts_per_trade=limits.get_int("sizing.max_contracts_per_trade"),
            max_premium_per_trade=limits.get_float("sizing.max_premium_per_trade"),
            max_premium_pct_of_equity=limits.get_float("sizing.max_premium_pct_of_equity"),
            min_contracts=limits.get_int("sizing.min_contracts"),
            assume_stop_gapped=limits.get_bool("sizing.assume_stop_gapped"),
            stop_pct=limits.get_float("exits.stop_pct"),
            max_concurrent_positions=limits.get_int("caps.max_concurrent_positions"),
            max_positions_per_symbol=limits.get_int("caps.max_positions_per_symbol"),
            max_open_premium=limits.get_float("caps.max_open_premium"),
            max_open_premium_pct_of_equity=limits.get_float("caps.max_open_premium_pct_of_equity"),
            max_entries_per_session=limits.get_int("caps.max_entries_per_session"),
            max_entries_per_symbol_per_session=limits.get_int(
                "caps.max_entries_per_symbol_per_session"
            ),
            size_multiplier_min=limits.get_float("agents.a3.size_multiplier_min"),
            size_multiplier_max=limits.get_float("agents.a3.size_multiplier_max"),
        )


@dataclass(frozen=True)
class SizingResult:
    """Every stage of the computation, kept for the decision log."""

    base_contracts: int
    model_multiplier: float
    multiplier_clamped: bool
    after_model: int
    final_contracts: int
    risk_budget: float
    risk_per_contract: float
    cost_per_contract: float
    total_cost: float
    total_max_risk: float
    gap_assumed: bool
    caps: tuple[CapVerdict, ...] = field(default_factory=tuple)
    binding_caps: tuple[str, ...] = ()
    rejected_reason: str | None = None

    @property
    def traded(self) -> bool:
        return self.final_contracts > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_contracts": self.base_contracts,
            "model_multiplier": self.model_multiplier,
            "multiplier_clamped": self.multiplier_clamped,
            "after_model": self.after_model,
            "final_contracts": self.final_contracts,
            "risk_budget": self.risk_budget,
            "risk_per_contract": self.risk_per_contract,
            "cost_per_contract": self.cost_per_contract,
            "total_cost": self.total_cost,
            "total_max_risk": self.total_max_risk,
            "gap_assumed": self.gap_assumed,
            "binding_caps": list(self.binding_caps),
            "rejected_reason": self.rejected_reason,
        }


def compute_size(
    cost_per_contract: float,
    max_risk_per_contract: float,
    account: AccountState,
    limits: SizingLimits,
    model_multiplier: float | None = None,
) -> SizingResult:
    """Size a position. Pure arithmetic over numbers -- no I/O, no model call.

    ``cost_per_contract`` is what one contract costs to open, in dollars,
    paying the ask. ``max_risk_per_contract`` is the most one contract can
    lose: the premium for a long option, the net debit for a debit vertical.
    Both are per-contract dollar figures with the 100x multiplier already
    applied -- see ``src.options.metrics``.
    """
    if cost_per_contract <= 0:
        return _rejected(
            f"cost_per_contract must be positive, got {cost_per_contract}",
            cost_per_contract, max_risk_per_contract, limits,
        )
    if max_risk_per_contract <= 0:
        return _rejected(
            f"max_risk_per_contract must be positive, got {max_risk_per_contract}",
            cost_per_contract, max_risk_per_contract, limits,
        )
    if account.equity <= 0:
        return _rejected("equity is not positive", cost_per_contract, max_risk_per_contract, limits)

    risk_budget = account.equity * limits.account_risk_pct_per_trade

    # The stop can be gapped through overnight, so the risk actually carried
    # per contract is the full premium unless explicitly configured otherwise.
    if limits.assume_stop_gapped:
        risk_per_contract = max_risk_per_contract
    else:
        risk_per_contract = max_risk_per_contract * min(1.0, abs(limits.stop_pct) / 100.0)
    if risk_per_contract <= 0:
        return _rejected(
            "risk_per_contract resolved to zero", cost_per_contract, max_risk_per_contract, limits
        )

    base_contracts = int(math.floor(risk_budget / risk_per_contract))

    # Agent 3's only lever, clamped on the way in. Shrink or veto, never enlarge.
    raw = 1.0 if model_multiplier is None else float(model_multiplier)
    multiplier = min(max(raw, limits.size_multiplier_min), limits.size_multiplier_max)
    multiplier_clamped = multiplier != raw
    after_model = int(math.floor(base_contracts * multiplier))

    # Structural guarantee of the monotone invariant.
    assert after_model <= base_contracts, "model multiplier increased size"

    final, verdicts = apply_caps(after_model, cost_per_contract, account, limits)
    assert final <= after_model, "caps increased size"

    binding = tuple(v.name for v in verdicts if v.binding)
    rejected: str | None = None
    if final < limits.min_contracts:
        rejected = (
            f"final size {final} below sizing.min_contracts {limits.min_contracts}"
            + (f" (bound by {', '.join(binding)})" if binding else "")
        )
        final = 0

    return SizingResult(
        base_contracts=base_contracts,
        model_multiplier=multiplier,
        multiplier_clamped=multiplier_clamped,
        after_model=after_model,
        final_contracts=final,
        risk_budget=risk_budget,
        risk_per_contract=risk_per_contract,
        cost_per_contract=cost_per_contract,
        total_cost=final * cost_per_contract,
        total_max_risk=final * max_risk_per_contract,
        gap_assumed=limits.assume_stop_gapped,
        caps=verdicts,
        binding_caps=binding,
        rejected_reason=rejected,
    )


def _rejected(
    reason: str, cost: float, max_risk: float, limits: SizingLimits
) -> SizingResult:
    return SizingResult(
        base_contracts=0, model_multiplier=0.0, multiplier_clamped=False, after_model=0,
        final_contracts=0, risk_budget=0.0, risk_per_contract=0.0,
        cost_per_contract=cost, total_cost=0.0, total_max_risk=0.0,
        gap_assumed=limits.assume_stop_gapped, caps=(), binding_caps=(),
        rejected_reason=reason,
    )
