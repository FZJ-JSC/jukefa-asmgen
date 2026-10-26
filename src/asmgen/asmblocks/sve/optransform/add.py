# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
SVE add (fadd)
"""

from typing import Any

from ....registers import asm_data_type as adt
from ...op import (
    optransform_modifier as mod,
    operand_modifier as opd_mod
)
from .base import sve_opd3_base
from .scalar import sve_scalar_add

class sve_add(sve_opd3_base):
    """
    SVE implementation of add (fadd)
    """
    scalar_class = sve_scalar_add
    # unpredicated fadd/fmul (vectors) exist; the predicated forms are
    # destructive: Zdn is both the destination and the first source
    unpredicated = True
    destructive = True
    widens = False

    inst_base = "add"

    def diagnose_failure(self, modifiers : set[mod],
                         operand_modifiers : dict[str,set[opd_mod]],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):
        super().diagnose_failure(modifiers, operand_modifiers, kwargs, dts)
        if mod.NP in modifiers:
            raise ValueError("NEON add has no NP-form")
