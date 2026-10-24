# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
X86_64 register types (scalar: GPRs and xmm scalar FP)
"""
from ...registers import greg_base, freg_base
from ...registers import asm_data_type as adt

#pylint: disable=too-few-public-methods
class x86_greg(greg_base):
    """
    x86_64 general purpose register
    """

    greg_names = [f'{i}' for i in \
            [str(j) for j in range(8,16)]+\
            ['a','b','c','d','si','di','bp','sp']]

    def __init__(self, reg_idx : int):
        self.reg_idx = reg_idx

    def name(self, size : int = 8) -> str:
        """
        Returns the correct name of the register for the given data size in bytes
        
        :param size: data size in bytes
        :type size: int
        :return: string containing the register name
        :rtype: str
        """
        prefixes = {1:'', 2: '', 4: 'e', 8: 'r'}

        num_suffixes = {1:'b', 2: 'w', 4: 'd', 8: ''}

        alpha_suffixes = {1:'l', 2: '', 4: '', 8: ''}

        name = x86_greg.greg_names[self.reg_idx]

        # a,b,c,d
        if self.reg_idx in [8,9,10,11]:
            if size > 1:
                name += 'x'

        if self.reg_idx < 8:
            name = 'r' + name + num_suffixes[size]
        else:
            name = prefixes[size] + name + alpha_suffixes[size]

        return name

    @property
    def idx(self) -> int:
        return self.reg_idx

    def __str__(self) -> str:
        return self.name()


class avx_freg(freg_base):
    """
    x86_64 scalar register (actually xmm)
    """
    def __init__(self, reg_idx : int):
        self.reg_idx = reg_idx

    @property
    def idx(self) -> int:
        return self.reg_idx

    def __str__(self) -> str:
        return f"xmm{self.idx}"


# type suffixes: vmovs{d,s,h}, vbroadcasts{d,s}, the generators' dt_suffixes
DT_SUFFIXES = {
        adt.DOUBLE : "d",
        adt.SINGLE : "s",
        adt.HALF   : "h",
        adt.SINT8  : "b",
        adt.SINT16 : "w",
        adt.SINT32 : "d",
        adt.SINT64 : "q",
        }


# element sizes' suffixes: vpbroadcast{b,w,d,q}, vpmull{w,d}
SIZE_SUFFIXES = {
        1 : 'b',
        2 : 'w',
        4 : 'd',
        8 : 'q',
        }
