# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RVV 1.0 and 0.7.1 multiplication
"""

from ...op.opd3 import opd3_modifier as mod

from .base import rvv_opd3_base
from ...riscv64.opd3 import riscv64_mul

class rvv_mul(rvv_opd3_base):
    """
    RVV 1.0 and 0.7.1 implementation of fma
    """
    scalar_class = riscv64_mul

    def get_base_inst(self, modifiers : set[mod]):
        return "mul"
