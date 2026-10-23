# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RISC-V 64-bit register types
"""
from ...registers import greg_base, freg_base
from ...registers import asm_data_type

#pylint: disable=too-few-public-methods

class riscv64_greg(greg_base):
    """
    RISC-V 64bit general purpose register
    """

    # TODO: this is ordered in such a way that the register tracker wouldn't accidentally
    #       allocate a reserved register. It's probably better to use
    #       0 = x0, 1 = x1, etc.. and develop a system where the special registers
    #       are automatically reserved (parameter to reg_tracker maybe?)
    names = [f't{i}' for i in range(7)] +\
            [f's{i}' for i in range(1,12)] +\
            [f'a{i}' for i in range(8)] +\
            ['ra', 'sp', 'gp', 'tp', 's0', 'zero']

    def __init__(self, reg_idx : int):
        self.reg_idx = reg_idx

    @property
    def idx(self) -> int:
        return self.reg_idx

    def __str__(self) -> str:
        return riscv64_greg.names[self.idx]

class riscv64_freg(freg_base):
    """
    RISC-V 64bit scalar register
    """

    names = [f'f{i}' for i in range(32)]

    def __init__(self, reg_idx : int):
        self.reg_idx = reg_idx

    @property
    def idx(self) -> int:
        return self.reg_idx

    def __str__(self) -> str:
        return riscv64_freg.names[self.idx]


# floating-point instructions' type suffixes: fcdt (fmv.d, fcvt.d.w, fadd.s)
# and fdt (the integer side of a move of bits: fmv.d.x, fmv.w.x)
FCDT_SUFFIXES = {
        asm_data_type.DOUBLE : "d",
        asm_data_type.SINGLE : "s",
        asm_data_type.HALF   : "h",
        }
FDT_SUFFIXES = {
        asm_data_type.DOUBLE : "d",
        asm_data_type.SINGLE : "w",
        asm_data_type.HALF   : "h",
        }

