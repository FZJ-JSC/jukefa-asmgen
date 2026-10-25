# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Operation on n data operands and 1 address operand and related structures
"""

import dataclasses
from enum import Enum,auto

from abc import abstractmethod

from .operation import operation
from .signature import operation_signature
from .modifier import operation_modifier
from .operand import operand_modifier
from .operand import operand_shape as osh, operand_type as ot, operand_role as orl
from .misc import make_ord_prefix

from ...registers import (
    asm_data_type as adt,
    data_reg,
    greg_base
)

class opmem_modifier(operation_modifier):
    """
    Possible modifiers for an instruction/operation
    """
    #ILANE = auto()   # select lane with an immediate
    #GLANE = auto()   # select lane with a greg
    TOFFSET = auto() # 2D offset in number of tiles
    VOFFSET = auto() # 1D offset in number of vectors
    IOFFSET = auto() # 1D offset in number of elements
    GOFFSET = auto() # 1D offset given by greg
    SCALEOFFSET = auto() # the greg offset scaled: scale_factor (x86: 1, 2, 4, 8)
    TINDEX = auto()  # 2D tile contains per-element indices/offsets
    VINDEX = auto()  # 1D vector contains per-element indices/offsets
    GSTRIDE = auto() # stride between elements given by greg
    ISTRIDE = auto() # stride between elements given by immediate
    POSTINC = auto() # increment address greg after operation
    STRUCT = auto()  # load a structure with multiple components
                     # (i.e [Re,Im], [x,y,z] or [r,g,b,a])
    #BCAST = auto()   # Broadcast one value into all lanes
    MASK  = auto()   # Masked operation
    #ROW = auto()     # Row of a treg
    #COL = auto()     # Column of a treg
    NT  = auto()     # Non-temporal ld/st
    PARAM = auto()   # The memory is an inline asm operand %[param] (inline asm only): no address register

class opmem_action(Enum):
    """
    Possible opmem actions
    """
    LOAD = auto()
    STORE = auto()
    PREFETCH = auto()

class opmem(operation):
    """
    Assembly/IR instruction with n data operand and 1 address operand

    Absraction for loads/stores/prefetches
    """
    NIE_MESSAGE="Method not implemented"

    def __call__(self, *, dregs : list[data_reg]|None = None,
                 areg : greg_base|None = None, dt : adt|None = None,
                 modifiers : set[opmem_modifier] = None,
                 operand_modifiers : dict[str,set[operand_modifier]] = None,
                 **kwargs) -> str:
        """
        Return the ASM/IR instruction

        :param dregs : Data registers; may be empty/omitted for operations
            without data operands (prefetches, cache maintenance)
        :type dregs : list[class:`asmgen.registers.data_reg`]
        :param areg : Address register
        :type areg : class:`asmgen.registers.greg_base`
        :param dt : Data type; required iff dregs are provided
        :type dt : class:`asmgen.registers.asm_data_type`
        :return : ASM/IR instruction corresponding to the operation
        :rtype : str
        """

        if modifiers is None:
            modifiers = set()
        if operand_modifiers is None:
            operand_modifiers = dict()

        if dregs and dt is None:
            raise ValueError("dt is required when dregs are provided")
        dregs = dregs if dregs else []

        # dts will be present in kwargs if an opmem routes to another opmem
        # like when a vector opmem routes to a scalar opmem
        # Since dts are rebuilt, drop it from kwargs
        kwargs.pop('dts', None)

        return self.execute(
            dregs=dregs,
            gregs=[areg] if areg is not None else [],
            dts={
                make_ord_prefix(i)+'dreg' : dt for i in range(len(dregs))
            },
            modifiers=modifiers,
            operand_modifiers=operand_modifiers,
            **kwargs
        )

    @abstractmethod
    # pylint: disable-next=arguments-differ
    def implementation(self, *, dregs : list[data_reg],
                       agreg : greg_base,
                       modifiers : set[opmem_modifier],
                       operand_modifiers : dict[str,set[operand_modifier]],
                       **kwargs) -> str:
        """
        opmem implementation/call interface

        Data types arrive via kwargs (`a_dt`, `b_dt`, ...) and are absent
        entirely when there are no dregs (prefetches, cache maintenance)
        """
        raise NotImplementedError(self.NIE_MESSAGE)


class dummy_opmem(opmem):
    """
    Dummy opd1a1 operation; ISAs assign this by default to operations they do not support
    """
    def all_signatures(self) -> list[operation_signature]:
        raise NotImplementedError(self.NIE_MESSAGE)

    def implementation(self, *,
                       dregs : list[data_reg], agreg : greg_base,
                       modifiers : set[opmem_modifier],
                       operand_modifiers : dict[str,set[operand_modifier]],
                       **kwargs) -> str:
        raise NotImplementedError(self.NIE_MESSAGE)


def param_signatures(signatures : list[operation_signature]) -> list[operation_signature]:
    """
    The PARAM forms of plain loads and stores: the same data operands, the
    memory an inline asm operand %[param] instead of an address register's
    (inline asm only)

    :param signatures: an opmem's signatures; those without modifiers that
        have a data and an address register get a PARAM form
    :return: the PARAM forms
    """
    out = []
    for s in signatures:
        if s.modifiers or 'agreg' not in s.operands or 'adreg' not in s.operands:
            continue
        operands = {k: v for k, v in s.operands.items() if k != 'agreg'}
        operands['param'] = osh(ot.NAME, orl.PARAM, None, None)
        out.append(dataclasses.replace(s, modifiers={opmem_modifier.PARAM}, operands=operands))
    return out

