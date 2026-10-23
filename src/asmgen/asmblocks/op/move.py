# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Operation moving data between data registers
"""

from enum import auto

from abc import abstractmethod

from .operation import operation
from .signature import operation_signature
from .modifier import operation_modifier
from .operand import operand_modifier
from .misc import make_ord_prefix

from ...registers import (
    asm_data_type as adt,
    data_reg,
)

# Modifiers are mostly on operands (ROW,LANE, etc...)
class move_modifier(operation_modifier):
    """
    Possible modifiers for an instruction/operation
    """
    MASK = auto()
    MULTIPLE_IN = auto()
    MULTIPLE_OUT = auto()
    BITS = auto()      # the bits as they are, between register files (greg to freg): no conversion


class move(operation):
    """
    Assembly/IR instruction moving data between data registers

    Absraction for moves,lane extracts/inserts, row/col extracts/inserts,
    bcasts, converts/casts, sign extends, etc...
    """
    NIE_MESSAGE="Method not implemented"

    def __call__(self, *,
                 dregs : list[data_reg],
                 dts : list[adt],
                 modifiers : set[move_modifier] = None,
                 operand_modifiers : dict[str,set[operand_modifier]] = None,
                 **kwargs) -> str:
        """
        Return the ASM/IR instruction
        
        :param dregs : Data registers
        :type dregs : list[class:`asmgen.registers.data_reg`]
        :param areg : Address register
        :type areg : class:`asmgen.registers.greg_base`
        :param dt : Data type
        :type dt : class:`asmgen.registers.asm_data_type`
        :return : ASM/IR instruction corresponding to the operation
        :rtype : str
        """

        if modifiers is None:
            modifiers = set()
        if operand_modifiers is None:
            operand_modifiers = dict()

        return self.execute(
            dregs=dregs,
            gregs=[],
            dts={
                make_ord_prefix(i)+'dreg' : dt for i,dt in zip(range(len(dregs)), dts)
            },
            modifiers=modifiers,
            operand_modifiers=operand_modifiers,
            **kwargs
        )

    @abstractmethod
    # pylint: disable-next=arguments-differ
    def implementation(self, *, dregs : list[data_reg],
                       dts : dict[str, adt],
                       modifiers : set[move_modifier],
                       operand_modifiers : dict[str,set[operand_modifier]],
                       **kwargs) -> str:
        """
        move implementation/call interface
        """
        raise NotImplementedError(self.NIE_MESSAGE)


class dummy_move(move):
    """
    Dummy move operation; ISAs assign this by default to operations they do not support
    """

    def get_signatures(self) -> list[operation_signature]:
        raise NotImplementedError(self.NIE_MESSAGE)

    def implementation(self, *, dregs : list[data_reg],
                       dts : dict[str, adt],
                       modifiers : set[move_modifier],
                       operand_modifiers : dict[str,set[operand_modifier]],
                       **kwargs) -> str:
        raise NotImplementedError(self.NIE_MESSAGE)
