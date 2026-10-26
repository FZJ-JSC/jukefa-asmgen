# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
NEON/ASIMD fma instruction
"""

from typing import Any

from ....registers import (
    asm_data_type as adt,
)
from .base import neon_opd3_base

from ...op import (
    optransform_modifier as mod,
    operand_modifier as opd_mod
)

class neon_fma(neon_opd3_base):
    """
    NEON/ASIMD implementation of fma
    """

    inst_base = "ml"
    supports_np = True
    has_acc_suffix = True

    def diagnose_failure(self, modifiers : set[mod],
                         operand_modifiers : dict[str,set[opd_mod]],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):
        super().diagnose_failure(modifiers, operand_modifiers, kwargs, dts)
        if mod.MASK in modifiers:
            raise ValueError("NEON mul has no masked form")
