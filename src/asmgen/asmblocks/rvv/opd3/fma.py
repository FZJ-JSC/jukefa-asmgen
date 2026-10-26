# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RVV 1.0 and 0.7.1 fused-multiply-accumulate
"""

from typing import Callable

from ...op.optransform import optransform_modifier as mod

from .base import rvv_opd3_base

class rvv_fma(rvv_opd3_base):
    """
    RVV 1.0 and 0.7.1 implementation of fma
    """

    supports_np = True

    def __init__(self,
                 asmwrap : Callable[[str],str]):
        super().__init__(asmwrap=asmwrap)

        self.operand_order = [2,1,0]

    def get_base_inst(self, modifiers : set[mod]):
        return "nmsac" if mod.NP in modifiers else "macc"
