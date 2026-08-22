"""Hard caps, expressed as a maximum contract count each.

Every cap answers the same question -- "how many contracts does this rule
allow?" -- so the final size is simply the minimum across all of them. That
uniformity is what makes the property test meaningful: there is one place
where size is decided and one comparison that decides it.

**Caps always win.** They are applied *after* any model output, never before,
and nothing downstream may raise the result. When a cap binds, both the
requested and the applied value are recorded so the decision log shows the
override rather than just the outcome.

A cap that can never bind is a bug, not a safety margin -- it means the real
constraint is somewhere we are not watching, and the first time it matters we
find out via a broker rejection on the entry path. :func:`audit_caps` exists
to catch exactly that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

__all__ = [
    "CapVerdict",
    "UNLIMITED",
    "apply_caps",
    "audit_caps",
    "contracts_affordable",
]

# A cap that does not constrain contract count on this particular trade -- a
# gate cap that passed, for instance. Never used as a final size.
UNLIMITED = math.inf


@dataclass(frozen=True)
class CapVerdict:
    """One cap's answer, and whether it was the one that bound."""

    name: str
    allowed_contracts: float
    limit_value: float
    observed: float
    stage: str
    binding: bool = False

    @property
    def is_gate(self) -> bool:
        """A gate cap permits 0 or everything -- position counts, entry counts."""
        return self.allowed_contracts in (0.0, UNLIMITED)


def contracts_affordable(budget: float, cost_per_contract: float) -> float:
    """How many whole contracts fit in ``budget``. Always rounds **down**.

    Rounding up by even one contract on a cap is how a limit silently becomes
    a suggestion.
    """
    if cost_per_contract <= 0:
        return UNLIMITED
    if budget <= 0:
        return 0.0
    return float(math.floor(budget / cost_per_contract))


def apply_caps(
    requested: int,
    cost_per_contract: float,
    account: Any,
    limits: Any,
) -> tuple[int, tuple[CapVerdict, ...]]:
    """Reduce ``requested`` to what every cap permits.

    ``account`` supplies live state -- equity, options buying power, open
    premium, position and entry counts. ``limits`` is a
    :class:`~src.risk.sizing.SizingLimits`.

    Returns the final count and every cap's verdict, including the ones that
    did not bind, because "which caps were checked" is part of the audit trail.
    """
    equity = account.equity
    verdicts: list[CapVerdict] = [
        CapVerdict(
            name="max_contracts_per_trade",
            allowed_contracts=float(limits.max_contracts_per_trade),
            limit_value=float(limits.max_contracts_per_trade),
            observed=float(requested),
            stage="sizing",
        ),
        CapVerdict(
            name="max_premium_per_trade",
            allowed_contracts=contracts_affordable(limits.max_premium_per_trade, cost_per_contract),
            limit_value=limits.max_premium_per_trade,
            observed=requested * cost_per_contract,
            stage="sizing",
        ),
        CapVerdict(
            name="max_premium_pct_of_equity",
            allowed_contracts=contracts_affordable(
                equity * limits.max_premium_pct_of_equity, cost_per_contract
            ),
            limit_value=equity * limits.max_premium_pct_of_equity,
            observed=requested * cost_per_contract,
            stage="sizing",
        ),
        # Live, not a static assumption. This is the cap that actually stops a
        # broker rejection, and it moves during the session as positions open.
        CapVerdict(
            name="options_buying_power",
            allowed_contracts=contracts_affordable(
                account.options_buying_power, cost_per_contract
            ),
            limit_value=account.options_buying_power,
            observed=requested * cost_per_contract,
            stage="entry",
        ),
        CapVerdict(
            name="max_open_premium",
            allowed_contracts=contracts_affordable(
                limits.max_open_premium - account.open_premium, cost_per_contract
            ),
            limit_value=limits.max_open_premium,
            observed=account.open_premium + requested * cost_per_contract,
            stage="portfolio",
        ),
        CapVerdict(
            name="max_open_premium_pct_of_equity",
            allowed_contracts=contracts_affordable(
                equity * limits.max_open_premium_pct_of_equity - account.open_premium,
                cost_per_contract,
            ),
            limit_value=equity * limits.max_open_premium_pct_of_equity,
            observed=account.open_premium + requested * cost_per_contract,
            stage="portfolio",
        ),
        _gate(
            "max_concurrent_positions",
            account.open_positions, limits.max_concurrent_positions, "portfolio",
        ),
        _gate(
            "max_positions_per_symbol",
            account.positions_in_symbol, limits.max_positions_per_symbol, "portfolio",
        ),
        _gate(
            "max_entries_per_session",
            account.entries_this_session, limits.max_entries_per_session, "entry",
        ),
        _gate(
            "max_entries_per_symbol_per_session",
            account.entries_this_symbol_this_session,
            limits.max_entries_per_symbol_per_session,
            "entry",
        ),
    ]

    allowed = min(v.allowed_contracts for v in verdicts)
    final = int(min(float(requested), allowed))
    final = max(0, final)

    # Mark every cap that produced the binding value, not just the first --
    # two caps landing on the same number is useful signal when tuning.
    marked = tuple(
        CapVerdict(
            name=v.name, allowed_contracts=v.allowed_contracts, limit_value=v.limit_value,
            observed=v.observed, stage=v.stage,
            binding=(v.allowed_contracts == allowed and allowed < requested),
        )
        for v in verdicts
    )
    return final, marked


def _gate(name: str, observed: int, limit: int, stage: str) -> CapVerdict:
    """A count cap: already at the limit means zero, otherwise no constraint."""
    return CapVerdict(
        name=name,
        allowed_contracts=0.0 if observed >= limit else UNLIMITED,
        limit_value=float(limit),
        observed=float(observed),
        stage=stage,
    )


def audit_caps(
    limits: Any,
    equity: float,
    options_buying_power: float,
    cost_per_contract: float,
) -> dict[str, Any]:
    """Report which caps can never bind given real account state.

    A cap set above what the account can fund is not conservative -- it is
    inert. The real constraint then lives at the broker, and the first time it
    matters the entry path takes a rejection instead of a clean pre-trade veto.

    This is not hypothetical: a dev account with $56,756 equity but only
    $530.72 of options buying power made `max_premium_per_trade: 1500` and
    `max_open_premium: 5000` both unreachable. Every one of our own caps would
    have passed a trade the broker then refused.
    """
    spendable = min(options_buying_power, equity)
    checks = {
        "max_premium_per_trade": limits.max_premium_per_trade,
        "max_premium_pct_of_equity": equity * limits.max_premium_pct_of_equity,
        "max_open_premium": limits.max_open_premium,
        "max_open_premium_pct_of_equity": equity * limits.max_open_premium_pct_of_equity,
    }
    unbindable = {
        name: {"cap": value, "spendable": spendable}
        for name, value in checks.items()
        if value > spendable
    }

    max_by_contracts = limits.max_contracts_per_trade * cost_per_contract
    if cost_per_contract > 0 and max_by_contracts > spendable:
        unbindable["max_contracts_per_trade"] = {
            "cap": max_by_contracts, "spendable": spendable,
        }

    return {
        "equity": equity,
        "options_buying_power": options_buying_power,
        "spendable": spendable,
        "cost_per_contract": cost_per_contract,
        "unbindable": unbindable,
        "all_caps_can_bind": not unbindable,
    }


def binding_names(verdicts: Iterable[CapVerdict]) -> tuple[str, ...]:
    return tuple(v.name for v in verdicts if v.binding)
