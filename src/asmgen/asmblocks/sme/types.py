# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
SME register types
"""

from ...registers import adt_size, treg_base
from ...registers import asm_data_type as adt


# pylint: disable-next=too-few-public-methods
class sme_treg(treg_base):
    """
    SME tile register
    """
    def __init__(self, reg_idx : int, dt : adt):
        # FP64,I64 : 8
        # FP32,I32 : 4
        # FP16,I16 : 2
        # FP8,I8 : 1
        max_tiles = adt_size(dt)

        if reg_idx > max_tiles:
            raise ValueError(f"SME has no tile {reg_idx} for data type {dt}")

        self.reg_idx = reg_idx

    @property
    def idx(self) -> int:
        return self.reg_idx

    def __str__(self) -> str:
        return f"za{self.idx}"
