# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
data move operation for RVV
"""

from typing import Callable

from ....registers import asm_data_type as adt, data_reg
from ...op.signature import operation_signature
from ...op.operand import operand_modifier as omod
from ...op.move import move, move_modifier as mod
from ...riscv64.types import riscv64_greg
from .signatures import make_rvv_move_signatures


class rvv_move(move):
    """
    RVV data move instructions: vmv, vfmv, vrgather
    """

    def __init__(self, asmwrap : Callable[[str], str]):
        self.asmwrap = asmwrap
        self.signatures = make_rvv_move_signatures()

    def get_signatures(self) -> list[operation_signature]:
        return self.signatures

    def implementation(self, *, dregs : list[data_reg],
                       dts : dict[str, adt],
                       modifiers : set[mod],
                       operand_modifiers : dict[str, set[omod]],
                       **kwargs) -> str:
        src, dst = dregs[0], dregs[1]
        amods = operand_modifiers.get('adreg', set())
        bmods = operand_modifiers.get('bdreg', set())
        integer = isinstance(src, riscv64_greg)
        if omod.ILANESHIFT in amods:
            if omod.FILLOTHERS in amods:
                floating = dts['bdreg'].name in ("DOUBLE", "SINGLE", "HALF")
                inst = "vfslide1down.vf" if floating else "vslide1down.vx"
                return self.asmwrap(f"{inst} {dst},{src},{kwargs['fillreg']}")
            return self.asmwrap(f"vslidedown.vi {dst},{src},{-kwargs['adreg_irot']}")
        if omod.ILANE in amods:
            if str(src) == str(dst):
                raise ValueError("vrgather.vi: the destination must not be the source")
            return self.asmwrap(f"vrgather.vi {dst},{src},{kwargs['adreg_lane']}")
        if omod.BCAST in bmods:
            return self.asmwrap(f"{'vmv.v.x' if integer else 'vfmv.v.f'} {dst},{src}")
        if omod.ILANE in bmods:
            return self.asmwrap(f"{'vmv.s.x' if integer else 'vfmv.s.f'} {dst},{src}")
        return self.asmwrap(f"vmv.v.v {dst},{src}")
