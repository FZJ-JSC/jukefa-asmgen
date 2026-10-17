# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
NEON/ASIMD opd3 base case
"""

from typing import Callable

from ....registers import (
    adt_size,
    asm_data_type as adt,
)

from ...aarch64.opd3.base import aarch64_simd_opd3_base
from ...op import operand_modifier as opd_mod

from .signatures import make_neon_opd3_signatures

class neon_opd3_base(aarch64_simd_opd3_base):
    """
    NEON/ASIMD base opd3 implementation with methods shared by all
    opd3 operations
    """

    inst_base = "invalid"

    supports_np = False

    def __init__(self,
                 asmwrap : Callable[[str],str],
                 dt_suffixes : dict[adt,str],
                 dt_idxsuffixes : dict[adt,str]
                 ):
        super().__init__(asmwrap,dt_suffixes,dt_idxsuffixes)

        self.signatures = make_neon_opd3_signatures(self.supports_np)

    def partial_inst_suffix(self, ways : int, part : int, a_dt=None, c_dt=None, **_):
        """
        FP16 to FP32 widens a register's lower half (fmlal) or its upper half
        (fmlal2); FP8 and BF16 take bottom and top (even and odd lanes), as
        on SVE: fmlalb/fmlalt, fmlallbb...fmlalltt, bfmlalb/bfmlalt
        """
        if (adt.FP16, adt.FP32) == (a_dt, c_dt):
            return "l" if 0 == part else "l2"
        return super().partial_inst_suffix(ways=ways, part=part)

    def build_operands(self, *,
                       adreg, bdreg, cdreg, a_dt, b_dt, c_dt,
                       operand_modifiers, **kwargs) -> tuple[str,str,str]:
        a, b, c = super().build_operands(adreg=adreg, bdreg=bdreg, cdreg=cdreg,
                                         a_dt=a_dt, b_dt=b_dt, c_dt=c_dt,
                                         operand_modifiers=operand_modifiers, **kwargs)
        if (adt.FP16, adt.FP32) == (a_dt, c_dt):
            # halves: the sources are named with the destination's lane count
            # (fmlal v0.4s, v1.4h, v2.4h), the half chosen by the instruction
            lanes = self.dt_suffixes[c_dt][:-1]
            a = f"{adreg}.{lanes}{self.dt_idxsuffixes[a_dt]}"
            if not {opd_mod.ILANE, opd_mod.BLOCKLANE} & operand_modifiers.get('bdreg', set()):
                b = f"{bdreg}.{lanes}{self.dt_idxsuffixes[b_dt]}"
        return a, b, c

