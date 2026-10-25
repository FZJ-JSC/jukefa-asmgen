# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# ------------------------------------------------------------------------------
"""
RVV 0.7.1's moves (0.6, phase 5): RVV's, but element 0 into a general
purpose register is vext.x.v (vmv.x.s is 1.0's), and 0.7.1 has no FP slides
(vfslide1down.vf, vfslide1up.vf) for its rotations through a scalar
"""
from __future__ import annotations

from ..op.operand import operand_modifier as opd_mod
from ..op.signature import operation_signature
from ..rvv.move import rvv_move

_FLOATS = ("DOUBLE", "SINGLE", "HALF")


class rvv071_move(rvv_move):
    """
    RVV 0.7.1 data moves
    """

    def element0(self, dst, src, floating : bool) -> str:
        if floating:
            return super().element0(dst, src, floating)
        return f"vext.x.v {dst},{src},zero"

    def get_signatures(self) -> list[operation_signature]:
        def fp_slide(s):
            return any(opd_mod.FILLOTHERS in (shape.modifiers or set())
                       and shape.dt is not None and shape.dt.name in _FLOATS
                       for shape in s.operands.values())
        return [s for s in super().get_signatures() if not fp_slide(s)]
