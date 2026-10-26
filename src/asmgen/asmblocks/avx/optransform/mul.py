# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
AVX mul (vmul)
"""

from typing import Any

from ...op import (
    optransform_modifier as mod,
    operand_modifier as opd_mod
)
from ....registers import asm_data_type as adt

from .base import avx_opd3_base
from ...x86.optransform import x86_mul

class avx_mul(avx_opd3_base):
    """
    AVX implementation of mul (vmul)
    """
    scalar_class = x86_mul

    def get_base_inst(self, modifiers : set[mod]):
        return "vmul"

    def diagnose_failure(self, modifiers: set[mod],
                         operand_modifiers : dict[str,set[opd_mod]],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):
        super().diagnose_failure(modifiers, operand_modifiers, kwargs, dts)
        if mod.NP in modifiers:
            raise ValueError("AVX mul has no NP form")
