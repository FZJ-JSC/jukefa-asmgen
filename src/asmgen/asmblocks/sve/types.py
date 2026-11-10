# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
SVE register types
"""
from ...registers import mreg_base, vreg_base


#pylint: disable=too-few-public-methods
class sve_vreg(vreg_base):
    """
    SVE vector register
    """
    def __init__(self, reg_idx : int):
        self.reg_idx = reg_idx

    @property
    def idx(self) -> int:
        return self.reg_idx

    def __str__(self) -> str:
        return f"z{self.idx}"

class sve_preg(mreg_base):
    """
    SVE predicate register
    """
    def __init__(self, reg_idx : int, is_pn : bool = False):
        self.reg_idx = reg_idx
        self.is_pn = is_pn

    @property
    def idx(self) -> int:
        return self.reg_idx

    def __str__(self) -> str:
        return f"pn{self.idx}" if self.is_pn else f"p{self.idx}"
