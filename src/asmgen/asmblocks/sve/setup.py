# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# ------------------------------------------------------------------------------
"""
SVE's setup: its all-true governing predicate
"""
from __future__ import annotations

from ..op.opset import opset_modifier
from ..setup import isa_setup, predication, setup_kind, setup_step


class sve_setup(isa_setup):
    """
    SVE: a governing predicate with every lane active, in the predicate
    register chosen (p0 by default), reserved and aliased as 'all_true' in the
    register tracker
    """

    ALIAS = "all_true"

    def __init__(self, gen, predicate : int = 0):
        super().__init__(gen)
        self.predicate = predicate

    def steps(self) -> list[setup_step]:
        return [setup_step(setup_kind.PREDICATE,
                           f"all-true governing predicate in p{self.predicate} for "
                           f"full-vector instructions")]

    def predication(self) -> predication:
        return predication.EXPLICIT

    def _index(self, rt) -> int:
        aliased = rt.aliased_regs.get("mreg", {}) if rt else {}
        return aliased.get(self.ALIAS, self.predicate)

    def setup(self, *, rt, dt, vl : int | None = None, engine : bool = True) -> str:
        super().setup(rt=rt, dt=dt, vl=vl)
        aliased = rt.aliased_regs.get("mreg") if rt else None
        if aliased is not None and self.ALIAS not in aliased:
            rt.reserve_specific_reg("mreg", self.predicate)
            rt.alias_reg("mreg", self.ALIAS, self.predicate)
        return self.gen.set(dreg=self.gen.mreg(self._index(rt)), dt=dt, modifiers={opset_modifier.CONSTMASK},
                            mask="alltrue")

    def all_true_predicate(self, *, rt, dt):
        return self.gen.mreg(self._index(rt))
