# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RVV 1.0 and 0.7.1 register types
"""
from ...registers import vreg_base


#pylint: disable=too-few-public-methods
class rvv_vreg(vreg_base):
    """
    RVV vector register
    """
    def __init__(self, reg_idx : int):
        self.reg_idx = reg_idx

    @property
    def idx(self) -> int:
        return self.reg_idx

    def __str__(self) -> str:
        return f"v{self.idx}"
