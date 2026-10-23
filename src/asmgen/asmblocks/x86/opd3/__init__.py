"""
x86-64 scalar arithmetic on general registers: add, sub, mul, shift. The
instructions take two operands (the destination is a source): another
destination is a copy first, and a destination that is the second source
is handled without losing it.
"""
from typing import Callable

from ...op.constraint import minmax_constraint
from ...op.opd3 import opd3_modifier
from ...op.scalar import scalar_arith
from ....registers import greg_base

IMM32 = [[minmax_constraint(minval=-2**31, maxval=2**31 - 1)]]


class x86_scalar_arith(scalar_arith):
    """
    The x86 operations' base: registers named through rpref
    """
    def __init__(self, *, asmwrap : Callable[[str], str], rpref : Callable):
        self.rpref = rpref
        super().__init__(asmwrap=asmwrap)

    def copy(self, *, src : str, dst : str) -> str:
        """
        dst = src first, unless they are one (src, dst: named registers)
        """
        return self.asmwrap(f"movq {src},{dst}") if src != dst else ""


class x86_add(x86_scalar_arith):
    """
    addq; a 32-bit signed immediate; lea for B shifted by 0 to 3
    """
    def imm_forms(self):
        return IMM32

    def shift_forms(self):
        # lea scales the index by 1, 2, 4 or 8; it has no subtraction
        return [([minmax_constraint(minval=0, maxval=3)], frozenset())]

    def with_register(self, *, adreg : greg_base, bdreg : greg_base, cdreg : greg_base, modifiers : set) -> str:
        pa, pb, pc = self.rpref(adreg), self.rpref(bdreg), self.rpref(cdreg)
        if pb == pc and pa != pc:
            return self.asmwrap(f"addq {pa},{pc}")
        return self.copy(src=pa, dst=pc) + self.asmwrap(f"addq {pb},{pc}")

    def with_immediate(self, *, adreg : greg_base, bimm : int, cdreg : greg_base, modifiers : set) -> str:
        pa, pc = self.rpref(adreg), self.rpref(cdreg)
        return self.copy(src=pa, dst=pc) + self.asmwrap(f"addq ${bimm},{pc}")

    def with_shifted(self, *, adreg : greg_base, bdreg : greg_base, bdreg_shift : int, cdreg : greg_base) -> str:
        return self.asmwrap(f"leaq ({self.rpref(adreg)},{self.rpref(bdreg)},{1 << bdreg_shift}),{self.rpref(cdreg)}")


class x86_sub(x86_scalar_arith):
    """
    subq; a 32-bit signed immediate
    """
    def imm_forms(self):
        return IMM32

    def with_register(self, *, adreg : greg_base, bdreg : greg_base, cdreg : greg_base, modifiers : set) -> str:
        pa, pb, pc = self.rpref(adreg), self.rpref(bdreg), self.rpref(cdreg)
        if pb == pc and pa != pc:
            return self.asmwrap(f"negq {pc}") + self.asmwrap(f"addq {pa},{pc}")
        return self.copy(src=pa, dst=pc) + self.asmwrap(f"subq {pb},{pc}")

    def with_immediate(self, *, adreg : greg_base, bimm : int, cdreg : greg_base, modifiers : set) -> str:
        pa, pc = self.rpref(adreg), self.rpref(cdreg)
        return self.copy(src=pa, dst=pc) + self.asmwrap(f"subq ${bimm},{pc}")


class x86_mul(x86_scalar_arith):
    """
    imulq; the immediate form takes three operands
    """
    def imm_forms(self):
        return IMM32

    def with_register(self, *, adreg : greg_base, bdreg : greg_base, cdreg : greg_base, modifiers : set) -> str:
        pa, pb, pc = self.rpref(adreg), self.rpref(bdreg), self.rpref(cdreg)
        if pb == pc and pa != pc:
            return self.asmwrap(f"imulq {pa},{pc}")
        return self.copy(src=pa, dst=pc) + self.asmwrap(f"imulq {pb},{pc}")

    def with_immediate(self, *, adreg : greg_base, bimm : int, cdreg : greg_base, modifiers : set) -> str:
        return self.asmwrap(f"imulq ${bimm},{self.rpref(adreg)},{self.rpref(cdreg)}")


class x86_shift(x86_scalar_arith):
    """
    shlq, shrq by an immediate, 0 to 63: opd3_modifier SHL or SHR. An amount
    in a register would have to be %cl, so that form isn't offered
    """
    def op_modifiers(self):
        return [{opd3_modifier.SHL}, {opd3_modifier.SHR}]

    def has_register_form(self):
        return False

    def imm_forms(self):
        return [[minmax_constraint(minval=0, maxval=63)]]

    def with_immediate(self, *, adreg : greg_base, bimm : int, cdreg : greg_base, modifiers : set) -> str:
        pa, pc = self.rpref(adreg), self.rpref(cdreg)
        inst = "shlq" if opd3_modifier.SHL in modifiers else "shrq"
        return self.copy(src=pa, dst=pc) + self.asmwrap(f"{inst} ${bimm},{pc}")
