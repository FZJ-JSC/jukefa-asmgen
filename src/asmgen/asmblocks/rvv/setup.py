# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# ------------------------------------------------------------------------------
"""
RVV's setup: vsetvli's element width, grouping and vector length
"""
from __future__ import annotations

from ...registers import asm_data_type as adt, adt_size
from ..op.opset import opset_modifier
from ..setup import isa_setup, predication, setup_kind, setup_step


class rvv_setup(isa_setup):
    """
    RVV: vsetvli sets the element width, the register grouping (LMUL) and the
    vector length, kept in a register. The vector length can be the most the
    hardware has, or fewer: on some implementations a shorter one performs
    better, and RVV instructions and address increments all use the length set.
    """

    def state(self, dt) -> dict[str, str]:
        """vsetvli's: the most elements, dt's element type"""
        return {"vlen": "max", "vtype": f"e{8 * adt_size(dt)}"}


    def steps(self) -> list[setup_step]:
        return [setup_step(setup_kind.VECTOR_LENGTH,
                           "vsetvli: element width, grouping and vector length, the "
                           "hardware's or fewer elements")]

    def supports_vl_override(self) -> bool:
        return True

    def predication(self) -> predication:
        return predication.IMPLICIT

    def element_type_is_state(self) -> bool:
        return True

    #: vsetvli's policies: tail and mask agnostic
    POLICIES = ", ta, ma"

    def _vsetvli(self, rd, rs, *, dt, group : int = 1) -> str:
        """
        vsetvli rd, rs: dt's element width; LMUL the generator's - a register
        the caller sees is a group of LMUL already, a wide one group of those;
        the vector length for rs elements (zero: the most) written to rd -
        both zero: the vector length kept, valid where SEW/LMUL stays the same,
        as from e32,m1 to e64,m2 for a widened group
        """
        return self.gen.asmwrap(f"vsetvli {rd}, {rs}, e{adt_size(dt) * 8}, "
                                f"m{self.gen.lmul * group}{self.POLICIES}")

    def setup_dt(self, *, dt, group : int = 1) -> str:
        return self._vsetvli("zero", "zero", dt=dt, group=group)

    @staticmethod
    def _aliased(rt, name: str) -> int:
        aliased = rt.aliased_regs.setdefault("greg", {})
        if name not in aliased:
            rt.alias_reg("greg", name, rt.reserve_any_reg("greg"))
        return aliased[name]

    def vector_length_register(self, *, rt):
        return self.gen.greg(self._aliased(rt, "vlen"))

    def setup(self, *, rt, dt, vl : int | None = None, engine : bool = True) -> str:
        super().setup(rt=rt, dt=dt, vl=vl)
        vlreg = self.vector_length_register(rt=rt)
        if vl is not None:
            avl = self.gen.greg(self._aliased(rt, "avl"))
            return self.gen.set(dreg=avl, dt=adt.UINT64, modifiers={opset_modifier.CONST}, value=vl) + \
                self._vsetvli(vlreg, avl, dt=dt)
        if "avl" in rt.aliased_regs.get("greg", {}):
            # the caller's vector length: a register it aliased avl and set -
            # a length preferred over the most there is (a multiple of 40
            # elements on some hardware: 240 doubles of 256), known at run
            # time only (BLIS's context)
            avl = self.gen.greg(rt.aliased_regs["greg"]["avl"])
            return self._vsetvli(vlreg, avl, dt=dt)
        return self._vsetvli(vlreg, "zero", dt=dt)
