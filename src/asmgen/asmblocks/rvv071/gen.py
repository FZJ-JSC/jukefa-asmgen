# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RISC-V RVV 0.7.1 asm generator and related types
"""

from ..rvv import rvv

from ...registers import asm_data_type as adt, adt_size, greg_base, vreg_base
from ...registers import asm_index_type as ait
from ..rvv.opset import rvv_opset
from .opmem import rvv071_load, rvv071_store
from .move import rvv071_move

# pylint: disable=too-many-public-methods
class rvv071(rvv):
    """
    RISC-V RVV 0.7.1 asmgen implementation
    """

    def __init__(self):
        super().__init__()
        # vsetvli without tail and mask policies
        self.set = rvv_opset(asmwrap=self.asmwrap, lmul_getter=lambda : self.lmul, policy=False)
        # 0.7.1's mnemonics (0.6, phase 5)
        self.load = rvv071_load(asmwrap=self.asmwrap, lmul_getter=lambda : self.lmul)
        self.store = rvv071_store(asmwrap=self.asmwrap, lmul_getter=lambda : self.lmul)
        self.move = rvv071_move(asmwrap=self.asmwrap)

    #: the mask population count
    vpopc = "vmpopc.m"

    dt_suffixes = {
            adt.DOUBLE  : "e",
            adt.SINGLE  : "w",
            adt.HALF    : "h",
            adt.FP8E4M3 : "b",
            adt.FP8E5M2 : "b",
            }

    def vsetvli(self, *, vlreg : greg_base, avlreg : greg_base, dt : adt) -> str:
        dt_size = 'e'+str(adt_size(dt)*8)
        return self.asmwrap(f"vsetvli {vlreg}, {avlreg}, {dt_size}, m1")

    def vsetvlmax(self, *, reg : greg_base, dt : adt) -> str:
        dt_size = 'e'+str(adt_size(dt)*8)
        return self.asmwrap(f"vsetvli {reg}, zero, {dt_size}, m1")
