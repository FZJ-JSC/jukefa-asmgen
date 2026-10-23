"""
Scalar arithmetic on general registers as operations (0.6, phase 2 of
docs/V06_REWORK.md): add, sub and mul of two registers, or of a register
and an immediate where the ISA has the form - its range a constraint. A
vector add or mul carries its base ISA's scalar operation as a member - its
signatures among its own, calls with general registers handed to it - as
rvv_opmem does with riscv64_opmem.

    gen.add(adreg=a, bdreg=b, cdreg=c, a_dt=UINT64, b_dt=UINT64, c_dt=UINT64)   c = a + b
    gen.add(adreg=a, imm=8, cdreg=c, a_dt=UINT64, c_dt=UINT64)                  c = a + 8
"""
from __future__ import annotations

from typing import Callable

from ...registers import asm_data_type as adt
from .opd3 import opd3
from .operand import operand_shape as osh, operand_type as ot, operand_role as orl, register_type as rt
from .operand import operand_modifier as opd_mod
from .signature import operation_signature

GREG_DTS = (adt.UINT64, adt.SINT64)


class scalar_arith(opd3):
    """
    An arithmetic operation on general registers: each ISA says which
    immediate forms it has (imm_forms) and writes the instructions
    """
    def __init__(self, *, asmwrap : Callable[[str], str]):
        self.asmwrap = asmwrap
        self._signatures = self.make_signatures()

    def imm_forms(self) -> list[list]:
        """
        The immediate forms' constraints, one list per form; none: no
        immediate form
        """
        return []

    def has_register_form(self) -> bool:
        """
        Whether B can be any general register (x86's shifts take only %cl)
        """
        return True

    def shift_forms(self) -> list[tuple[list, frozenset]]:
        """
        Forms adding a register shifted left first (operand modifier ISHIFT
        on bdreg, the amount bdreg_shift): the amount's constraints and the
        features each needs; none: no such form
        """
        return []

    def make_signatures(self) -> list[operation_signature]:
        sigs = []
        for dt in GREG_DTS:
            def reg(dt=dt, mods=None):
                return osh(ot.REGISTER, orl.DATA, rt.GP, dt, modifiers=set(mods or ()))
            if self.has_register_form():
                sigs.append(operation_signature(modifiers=set(), structural_params={},
                                                operands={"adreg": reg(), "bdreg": reg(), "cdreg": reg()}))
            for constraints, requires in self.shift_forms():
                amount = osh(ot.IMMEDIATE, orl.PARAM, None, None, value_constraints=list(constraints))
                sigs.append(operation_signature(modifiers=set(), structural_params={}, requires=frozenset(requires),
                                                operands={"adreg": reg(), "bdreg": reg(mods={opd_mod.ISHIFT}),
                                                          "cdreg": reg(), "bdreg_shift": amount}))
            for constraints in self.imm_forms():
                imm = osh(ot.IMMEDIATE, orl.PARAM, None, None, value_constraints=list(constraints))
                sigs.append(operation_signature(modifiers=set(), structural_params={},
                                                operands={"adreg": reg(), "cdreg": reg(), "imm": imm}))
        return sigs

    def get_signatures(self) -> list[operation_signature]:
        return self._signatures

    # pylint: disable-next=arguments-differ
    def implementation(self, *, adreg, cdreg, bdreg=None, imm=None, bdreg_shift=None, **_) -> str:
        if bdreg_shift is not None:
            return self.with_shifted(cdreg, adreg, bdreg, bdreg_shift)
        if imm is not None:
            return self.with_immediate(cdreg, adreg, imm)
        return self.with_register(cdreg, adreg, bdreg)

    def with_register(self, c, a, b) -> str:
        raise NotImplementedError(self.NIE_MESSAGE)

    def with_immediate(self, c, a, imm : int) -> str:
        raise NotImplementedError(self.NIE_MESSAGE)

    def with_shifted(self, c, a, b, shift : int) -> str:
        """
        c = a + (b << shift), or minus
        """
        raise NotImplementedError(self.NIE_MESSAGE)


def scalar_operands(kwargs : dict) -> dict:
    """
    What a vector operation's implementation hands on to its scalar member
    besides the named operands: anything but execute's bookkeeping
    """
    return {k: v for k, v in kwargs.items() if k not in ("dts", "dregs", "gregs")}

