# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Operation on 3 data operands and related structures
"""
from abc import abstractmethod
from enum import Enum,auto

from .operation import operation
from .modifier import operation_modifier
from .operand import operand_modifier
from .signature import operation_signature

from ...registers import asm_data_type as adt, data_reg

class widening_method(Enum):
    """
    Possible methods ISAs can have for widening instructions
    """
    NONE = auto()
    VEC_GROUP = auto()
    VEC_MULTI = auto()
    DOT_NEIGHBOURS = auto()
    # one instruction per part of the wide result, each part a register:
    SPLIT_CONTIGUOUS = auto()   # part p holds a contiguous run of the elements
                                # (NEON fmlal/fmlal2: the lower and upper half)
    SPLIT_INTERLEAVED = auto()  # part p holds every ways-th element from p
                                # (bottom/top: fmlalb even, fmlalt odd elements)

class opd3_modifier(operation_modifier):
    """
    Possible modifiers for an opd3 instruction/operation
    """
    NP = auto()       # FMA negate product c=-(a*b)+c
    NA = auto()       # FMA negate addend c=(a*b)-c
    NX = auto()       # FMA negate everything c=-(a*b)-c
    MULC = auto()     # FMA C is a multiplicand, B is an addend c=(a*c)+b
    #IDX = auto()      # lane-fma
    #BLOCKIDX = auto() # block-lane-fma (i.e SVE FMA selects a lane for
    #                  # each 128-bit block of elements)
    #REGIDX = auto()
    PART = auto()
    #VF = auto()
    MASK = auto()


class opd3(operation):
    """
    Assembly/IR instruction with 3 data operands

    Data operands means registers. Exanding to memory (shared/tensor mem in GPUs,
    TCMs, x86 mem operands, etc...) operands is planned
    Examples:
      fma      op1, op2, op3      : op3      <-   op1 * op2      + op3
      fma.np   op1, op2, op3      : op3      <- -(op1 * op2)     + op3
      fma.idx  op1, op2, op3, idx : op3      <-   op1 * op2[idx] + op3
      fmul     op1, op2, op3      : op3      <-   op1 * op2
      opa      op1, op2, op3      : op3      <-   op1 o op2      + op3
      dota     op1, op2, op3      : op3      <-   op1 . op2      + op3
      dota.idx op1, op2, op3      : op3[idx] <-   op1 . op2      + op3[idx]
      mma      op1, op2, op3      : op3      <-   op1 x op2      + op3
    (*: elementwise multiplication)
    (o: outer product)
    (.: dot product)
    (x: matrix product)
    """
    NIE_MESSAGE="Method not implemented"

    def __call__(self, *,
                 adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                 a_dt : adt, b_dt : adt, c_dt : adt,
                 modifiers : set[opd3_modifier] = None,
                 operand_modifiers : dict[str,set[operand_modifier]] = None,
                 **kwargs) -> str:
        """
        Return the ASM/IR instruction
        
        :param adreg : Data register containing elements of the A component
        :type adreg : class:`asmgen.registers.data_reg`
        :param bdreg : Data register containing elements of the B component
        :type bdreg : class:`asmgen.registers.data_reg`
        :param cdreg : Data register containing elements of the C component
        :type cdreg : class:`asmgen.registers.data_reg`
        :param a_dt : Data type of the A component
        :type a_dt : class:`asmgen.registers.asm_data_type`
        :param b_dt : Data type of the B component
        :type b_dt : class:`asmgen.registers.asm_data_type`
        :param c_dt : Data type of the C component
        :type c_dt : class:`asmgen.registers.asm_data_type`
        :return : ASM/IR instruction corresponding to the operation
        :rtype : str
        """

        if modifiers is None:
            modifiers = set()
        if operand_modifiers is None:
            operand_modifiers = dict()

        return self.execute(
            dregs=[adreg,bdreg,cdreg],
            gregs=[],
            dts={'adreg':a_dt,'bdreg':b_dt,'cdreg':c_dt},
            modifiers=modifiers,
            operand_modifiers=operand_modifiers,
            **kwargs
        )

    @abstractmethod
    # pylint: disable-next=arguments-differ
    def implementation(self, *,
                       adreg : data_reg,
                       bdreg : data_reg, cdreg : data_reg,
                       a_dt : adt, b_dt : adt, c_dt : adt,
                       modifiers : set[opd3_modifier] = None,
                       operand_modifiers : dict[str,set[operand_modifier]] = None,
                       **kwargs) -> str:
        """
        opd3 implementation/call interface
        """


class dummy_opd3(opd3):
    """
    Dummy opd3 operation; ISAs assign this by default to operations they do not support
    """

    def get_signatures(self) -> list[operation_signature]:
        raise NotImplementedError(self.NIE_MESSAGE)

    def implementation(self, *,
                       adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                       a_dt : adt, b_dt : adt, c_dt : adt,
                       modifiers : set[opd3_modifier],
                       operand_modifiers : dict[str,set[operand_modifier]] = None,
                       **kwargs) -> str:
        raise NotImplementedError(self.NIE_MESSAGE)
