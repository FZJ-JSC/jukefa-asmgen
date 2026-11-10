# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
NEON/ASIMD register types
"""
from ...registers import vreg_base


#pylint: disable=too-few-public-methods
class neon_vreg(vreg_base):
    """
    NEON/ASIMD vector register
    """
    def __init__(self, reg_idx : int):
        self.reg_idx = reg_idx
    def __str__(self) -> str:
        return f"v{self.idx}"

    @property
    def idx(self) -> int:
        return self.reg_idx
