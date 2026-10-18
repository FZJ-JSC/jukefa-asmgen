# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
NEON/ASIMD fmul instruction
"""

from typing import Any

from ....registers import (
    asm_data_type as adt
)
from ...op import opd3_modifier as mod,operand_modifier as opd_mod
from .base import neon_opd3_base

class neon_fmul(neon_opd3_base):
    """
    NEON/ASIMD implementation of fmul
    """
    widens = False

    inst_base = "mul"

    def diagnose_failure(self,
                         modifiers : set[mod],
                         operand_modifiers : dict[str,set[opd_mod]],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):
        super().diagnose_failure(modifiers, operand_modifiers, kwargs, dts)
        if mod.NP in modifiers:
            raise ValueError("NEON mul has no NP-form")
        if mod.MASK in modifiers:
            raise ValueError("NEON mul has no masked form")


    # Leave as info for when implementing signatures
    #def check_dts(self, dts : dict[str,adt]):
    #    super().check_dts(dts)

    #    if dts['cdreg'] in [adt.UINT64, adt.UINT32, adt.UINT16]:
    #        if adt_size(dts['adreg']) == adt_size(dts['cdreg']):
    #            raise ValueError("only widening variants exist for unsigned integer types")
