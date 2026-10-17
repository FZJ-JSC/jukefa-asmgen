# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
data move operation for NEON
"""

from typing import Callable

from ....registers import asm_data_type as adt, data_reg
from ...op.signature import operation_signature
from ...op.operand import operand_modifier as omod
from ...op.move import move, move_modifier as mod
from ...aarch64.types import aarch64_greg
from .signatures import make_neon_move_signatures


class neon_move(move):
    """
    NEON data move instructions: mov, dup, ins
    """

    def __init__(self, asmwrap : Callable[[str], str],
                 dt_suffixes : dict[adt, str], dt_idxsuffixes : dict[adt, str]):
        self.asmwrap = asmwrap
        self.dt_suffixes = dt_suffixes
        self.dt_idxsuffixes = dt_idxsuffixes
        self.signatures = make_neon_move_signatures()

    def get_signatures(self) -> list[operation_signature]:
        return self.signatures

    def implementation(self, *, dregs : list[data_reg],
                       dts : dict[str, adt],
                       modifiers : set[mod],
                       operand_modifiers : dict[str, set[omod]],
                       **kwargs) -> str:
        src, dst = dregs[0], dregs[1]
        dt = dts['bdreg']
        el = self.dt_idxsuffixes[dt]
        amods = operand_modifiers.get('adreg', set())
        bmods = operand_modifiers.get('bdreg', set())

        if omod.ILANE in amods:
            source = f"v{src.idx}.{el}[{kwargs['adreg_lane']}]"
        elif isinstance(src, aarch64_greg):
            source = str(src.retype(dt))
        elif amods or bmods:
            # a scalar FP register is lane 0 of its vector register
            source = f"v{src.idx}.{el}[0]"
        else:
            return self.asmwrap(f"mov v{dst.idx}.16b,v{src.idx}.16b")

        if omod.BCAST in bmods:
            return self.asmwrap(f"dup v{dst.idx}.{self.dt_suffixes[dt]},{source}")
        return self.asmwrap(f"ins v{dst.idx}.{el}[{kwargs['bdreg_lane']}],{source}")
