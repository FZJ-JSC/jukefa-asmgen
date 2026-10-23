"""
Operation setting a register to what comes from the instruction stream
alone (0.6, phase 4 of docs/V06_REWORK.md): a constant (CONST), a mask
(CONSTMASK), the vector length (VLEN), indices (INDEX).

    gen.set(dreg=v, dt=adt.DOUBLE, modifiers={opset_modifier.CONST}, constant="zero")

The register set is dreg, the operation's one output; forms with inputs
name them. A vector ISA's set carries its base ISA's for general and
floating-point registers as a member, as rvv_opmem carries riscv64_opmem.
"""
from __future__ import annotations

from abc import abstractmethod
from enum import auto
from typing import Callable

from .modifier import operation_modifier
from .operand import operand_shape as osh, operand_type as ot, operand_role as orl, register_type as rt
from .operation import operation
from .scalar import GREG_DTS, scalar_operands
from .signature import operation_signature
from ...registers import asm_data_type as adt, data_reg, greg_base, freg_base


class opset_modifier(operation_modifier):
    """
    What set sets
    """
    CONST = auto()      # a constant: constant="zero"
    CONSTMASK = auto()  # a mask: mask="alltrue" or "allfalse" (to come)
    VLEN = auto()       # the vector length: bytes, or elements of dt (to come)
    INDEX = auto()      # element indices times a stride (to come)


FREG_DTS = (adt.DOUBLE, adt.SINGLE, adt.HALF)
VREG_DTS = (adt.DOUBLE, adt.SINGLE, adt.HALF, adt.UINT64, adt.SINT64, adt.UINT32, adt.SINT32,
            adt.UINT16, adt.SINT16, adt.UINT8, adt.SINT8)


def zero_signatures(rtype, dts) -> list[operation_signature]:
    """
    CONST zero of a register of rtype, one signature per type
    """
    return [operation_signature(modifiers={opset_modifier.CONST}, structural_params={"constant": "zero"},
                                operands={"dreg": osh(ot.REGISTER, orl.DATA, rtype, dt)}) for dt in dts]


class opset(operation):
    """
    Assembly/IR instruction(s) setting dreg from the instruction stream
    """
    NIE_MESSAGE = "Method not implemented"

    # pylint: disable-next=arguments-differ
    def __call__(self, *, dreg, dt : adt, modifiers=None, operand_modifiers=None, **kwargs) -> str:
        return self.execute(dregs=[], dts={"dreg": dt}, modifiers=modifiers or set(),
                            operand_modifiers=operand_modifiers or {}, dreg=dreg, dt=dt, **kwargs)

    @abstractmethod
    # pylint: disable-next=arguments-differ
    def implementation(self, *, dreg, dt : adt, modifiers, operand_modifiers, **kwargs) -> str:
        raise NotImplementedError(self.NIE_MESSAGE)


class scalar_opset(opset):
    """
    set on a base ISA's general and floating-point registers: each ISA
    writes the instructions
    """
    def __init__(self, *, asmwrap : Callable[[str], str]):
        self.asmwrap = asmwrap
        self.signatures = self.make_signatures()

    def make_signatures(self) -> list[operation_signature]:
        """
        The ISA's signatures (its opset/signatures.py)
        """
        return zero_signatures(rt.GP, GREG_DTS) + zero_signatures(rt.FP, FREG_DTS)

    def get_signatures(self) -> list[operation_signature]:
        return self.signatures

    # pylint: disable-next=arguments-differ
    def implementation(self, *, dreg : data_reg, dt : adt, modifiers=None, constant=None, **_) -> str:
        if isinstance(dreg, greg_base):
            return self.zero_greg(dreg=dreg)
        return self.zero_freg(dreg=dreg, dt=dt)

    def zero_greg(self, *, dreg : greg_base) -> str:
        """
        A general register zeroed
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    def zero_freg(self, *, dreg : freg_base, dt : adt) -> str:
        """
        A floating-point register zeroed
        """
        raise NotImplementedError(self.NIE_MESSAGE)


class vector_opset(opset):
    """
    set on a vector ISA's registers; general and floating-point ones go to
    scalar_op, the base ISA's set
    """
    scalar_op : scalar_opset = None
    signatures : list = []

    def get_signatures(self) -> list[operation_signature]:
        return [*self.signatures, *(self.scalar_op.get_signatures() if self.scalar_op else [])]

    # pylint: disable-next=arguments-differ
    def implementation(self, *, dreg, dt, modifiers=None, operand_modifiers=None, constant=None, **kwargs) -> str:
        # If a scalar register is passed, forward to the base ISA's set
        if self.scalar_op is not None and isinstance(dreg, (greg_base, freg_base)):
            return self.scalar_op(dreg=dreg, dt=dt, modifiers=modifiers, operand_modifiers=operand_modifiers,
                                  constant=constant, **scalar_operands(kwargs))
        return self.zero_reg(dreg=dreg, dt=dt)

    def zero_reg(self, *, dreg : data_reg, dt : adt) -> str:
        """
        A vector register (or tile) zeroed
        """
        raise NotImplementedError(self.NIE_MESSAGE)
