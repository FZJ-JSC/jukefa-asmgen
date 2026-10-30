# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# ------------------------------------------------------------------------------
"""
SME's setup: streaming mode and ZA (smstart, smstop) on SVE's
"""
from __future__ import annotations

from ..op.opset import opset_modifier
from ..setup import setup_kind, setup_step
from ..sve.setup import sve_setup


class sme_setup(sve_setup):
    """
    SME: streaming mode with ZA on and off around the kernel, SVE's predicate,
    and an all-true predicate-as-counter for multi-vector instructions
    """

    def state(self, dt) -> dict[str, str]:
        """smstart's: streaming mode, ZA on"""
        return {**super().state(dt), "sm": "1", "za": "1"}

    def teardown_state(self) -> dict[str, str]:
        """smstop's: out of streaming mode, ZA off"""
        return {**super().teardown_state(), "sm": "0", "za": "0"}

    def engine(self) -> tuple[dict[str, str], dict[str, str]]:
        """smstart's and smstop's: streaming mode and ZA on, off"""
        return {"sm": "1", "za": "1"}, {"sm": "0", "za": "0"}


    COUNTER = 8

    def clobbers(self) -> dict[str, list[int]]:
        """
        smstart and smstop zero the vector registers: every FP register the
        function's caller may expect kept
        """
        return {"freg": list(range(self.gen.max_fregs()))}

    def counter(self) -> bool:
        """
        Whether the predicate-as-counter is set: an SME2 register - with SME2
        enabled, or no features chosen (every one available)
        """
        features = self.gen.features
        return features is None or "SME2" in features

    def steps(self) -> list[setup_step]:
        steps = [setup_step(setup_kind.ENGINE, "streaming SVE mode and ZA: smstart, smstop"),
                 *super().steps()]
        if self.counter():
            steps.append(setup_step(setup_kind.PREDICATE,
                                    f"all-true predicate-as-counter in pn{self.COUNTER} for "
                                    f"multi-vector instructions"))
        return steps

    def setup(self, *, rt, dt, vl : int | None = None, engine : bool = True) -> str:
        # smstart first: it zeroes the vector and predicate registers
        out = (self.gen.asmwrap("smstart") if engine else "") + super().setup(rt=rt, dt=dt, vl=vl)
        if self.counter():
            out += self.gen.set(dreg=self.gen.mreg(self.COUNTER, True), dt=dt, modifiers={opset_modifier.CONSTMASK},
                                mask="alltrue")
        return out

    def teardown(self, *, rt, dt, engine : bool = True) -> str:
        return super().teardown(rt=rt, dt=dt) + (self.gen.asmwrap("smstop") if engine else "")
