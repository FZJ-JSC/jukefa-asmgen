"""
x86-64 scalar arithmetic on general registers: add, sub, mul. The
instructions take two operands (the destination is a source): another
destination is a copy first, and a destination that is the second source
is handled without losing it.
"""
from typing import Callable

from ...op.constraint import minmax_constraint
from ...op.scalar import scalar_arith

IMM32 = [[minmax_constraint(minval=-2**31, maxval=2**31 - 1)]]


class x86_scalar_arith(scalar_arith):
    def __init__(self, *, asmwrap : Callable[[str], str], rpref : Callable):
        self.rpref = rpref
        super().__init__(asmwrap=asmwrap)

    def _copy(self, pa, pc) -> str:
        return self.asmwrap(f"movq {pa},{pc}") if pa != pc else ""


class x86_add(x86_scalar_arith):
    """
    addq; a 32-bit signed immediate
    """
    def imm_forms(self):
        return IMM32

    def with_register(self, c, a, b):
        pa, pb, pc = self.rpref(a), self.rpref(b), self.rpref(c)
        if pb == pc and pa != pc:
            return self.asmwrap(f"addq {pa},{pc}")
        return self._copy(pa, pc) + self.asmwrap(f"addq {pb},{pc}")

    def with_immediate(self, c, a, imm):
        pa, pc = self.rpref(a), self.rpref(c)
        return self._copy(pa, pc) + self.asmwrap(f"addq ${imm},{pc}")

    def shift_forms(self):
        # lea scales the index by 1, 2, 4 or 8; it has no subtraction
        return [([minmax_constraint(minval=0, maxval=3)], frozenset())]

    def with_shifted(self, c, a, b, shift):
        return self.asmwrap(f"leaq ({self.rpref(a)},{self.rpref(b)},{1 << shift}),{self.rpref(c)}")


class x86_sub(x86_scalar_arith):
    """
    subq; a 32-bit signed immediate
    """
    def imm_forms(self):
        return IMM32

    def with_register(self, c, a, b):
        pa, pb, pc = self.rpref(a), self.rpref(b), self.rpref(c)
        if pb == pc and pa != pc:
            return self.asmwrap(f"negq {pc}") + self.asmwrap(f"addq {pa},{pc}")
        return self._copy(pa, pc) + self.asmwrap(f"subq {pb},{pc}")

    def with_immediate(self, c, a, imm):
        pa, pc = self.rpref(a), self.rpref(c)
        return self._copy(pa, pc) + self.asmwrap(f"subq ${imm},{pc}")


class x86_mul(x86_scalar_arith):
    """
    imulq; the immediate form takes three operands
    """
    def imm_forms(self):
        return IMM32

    def with_register(self, c, a, b):
        pa, pb, pc = self.rpref(a), self.rpref(b), self.rpref(c)
        if pb == pc and pa != pc:
            return self.asmwrap(f"imulq {pa},{pc}")
        return self._copy(pa, pc) + self.asmwrap(f"imulq {pb},{pc}")

    def with_immediate(self, c, a, imm):
        return self.asmwrap(f"imulq ${imm},{self.rpref(a)},{self.rpref(c)}")


class x86_shift(x86_scalar_arith):
    """
    A shift by an immediate, 0 to 63: the amount in a register would have
    to be %cl, so that form isn't offered
    """
    mnemonic = None

    def has_register_form(self):
        return False

    def imm_forms(self):
        return [[minmax_constraint(minval=0, maxval=63)]]

    def with_immediate(self, c, a, imm):
        pa, pc = self.rpref(a), self.rpref(c)
        return self._copy(pa, pc) + self.asmwrap(f"{self.mnemonic} ${imm},{pc}")


class x86_shl(x86_shift):
    """
    shlq: logical shift left
    """
    mnemonic = "shlq"


class x86_shr(x86_shift):
    """
    shrq: logical shift right
    """
    mnemonic = "shrq"

