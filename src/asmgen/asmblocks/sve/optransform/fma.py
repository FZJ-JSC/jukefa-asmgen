# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
SVE fma instruction
"""

from .base import sve_opd3_base

class sve_fma(sve_opd3_base):
    """
    SVE implementation of fma
    """

    inst_base = "ml"
    supports_np = True
    has_acc_suffix = True
