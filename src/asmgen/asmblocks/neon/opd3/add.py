# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
NEON/ASIMD add (fadd)
"""

from typing import Any

from ....registers import (
    asm_data_type as adt,
)
from ...op import (
    opd3_modifier as mod,
    operand_modifier as opd_mod
)
from .base import neon_opd3_base

class neon_add(neon_opd3_base):
    """
    NEON/ASIMD implementation of add (fadd)
    """
    widens = False

    inst_base = "add"

    def diagnose_failure(self, modifiers : set[mod],
                         operand_modifiers : dict[str,set[opd_mod]],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):
        super().diagnose_failure(modifiers, operand_modifiers, kwargs, dts)
        if mod.NP in modifiers:
            raise ValueError("NEON add has no NP-form")
        if mod.MASK in modifiers:
            raise ValueError("NEON add has no masked form")
