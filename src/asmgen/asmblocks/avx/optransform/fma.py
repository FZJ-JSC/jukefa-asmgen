# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
AVX fma instruction
"""

from ...op import optransform_modifier as mod

from .base import avx_opd3_base

class avx_fma(avx_opd3_base):
    """
    AVX implementation of fma
    """

    def get_base_inst(self, modifiers : set[mod]):
        return "vfnmadd231" if mod.NP in modifiers else "vfmadd231"
