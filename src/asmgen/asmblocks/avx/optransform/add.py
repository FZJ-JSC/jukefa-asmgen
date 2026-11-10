# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
AVX add (vadd)
"""

from typing import Any

from ....registers import asm_data_type as adt
from ...op import operand_modifier as opd_mod
from ...op import optransform_modifier as mod
from ...x86.optransform import x86_add
from .base import avx_opd3_base


class avx_add(avx_opd3_base):
    """
    AVX implementation of add (vadd)
    """
    scalar_class = x86_add

    def get_base_inst(self, modifiers : set[mod]):
        return "vadd"

    def diagnose_failure(self, modifiers: set[mod],
                         operand_modifiers : dict[str,set[opd_mod]],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):
        super().diagnose_failure(modifiers, operand_modifiers, kwargs, dts)
        if mod.NP in modifiers:
            raise ValueError("AVX add has no NP form")
