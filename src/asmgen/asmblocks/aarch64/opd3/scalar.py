"""
AArch64 scalar arithmetic on general registers: add, sub, mul
"""
from ...op.constraint import anyof_constraint, minmax_constraint, multiple_constraint
from ...op.scalar import scalar_arith

SHIFTED_MAX = 4095 * 4096


def _imm_forms():
    # 12 bits, or 12 bits shifted left by 12; a negative value is the other
    # operation's (the assembler turns add #-8 into sub #8)
    return [[anyof_constraint(alternatives=[
        [minmax_constraint(minval=-4095, maxval=4095)],
        [multiple_constraint(multiple=4096), minmax_constraint(minval=-SHIFTED_MAX, maxval=SHIFTED_MAX)],
    ])]]


class aarch64_add(scalar_arith):
    """
    add
    """
    def imm_forms(self):
        return _imm_forms()

    def with_register(self, c, a, b):
        return self.asmwrap(f"add {c},{a},{b}")

    def with_immediate(self, c, a, imm):
        return self.asmwrap(f"add {c},{a},#{imm}")

    def shift_forms(self):
        return [([minmax_constraint(minval=0, maxval=63)], frozenset())]

    def with_shifted(self, c, a, b, shift):
        return self.asmwrap(f"add {c},{a},{b}, LSL #{shift}" if shift else f"add {c},{a},{b}")


class aarch64_sub(scalar_arith):
    """
    sub
    """
    def imm_forms(self):
        return _imm_forms()

    def with_register(self, c, a, b):
        return self.asmwrap(f"sub {c},{a},{b}")

    def with_immediate(self, c, a, imm):
        return self.asmwrap(f"sub {c},{a},#{imm}")

    def shift_forms(self):
        return [([minmax_constraint(minval=0, maxval=63)], frozenset())]

    def with_shifted(self, c, a, b, shift):
        return self.asmwrap(f"sub {c},{a},{b}, LSL #{shift}" if shift else f"sub {c},{a},{b}")


class aarch64_mul(scalar_arith):
    """
    mul; no immediate form
    """
    def with_register(self, c, a, b):
        return self.asmwrap(f"mul {c},{a},{b}")


class aarch64_shl(scalar_arith):
    """
    lsl: logical shift left, by 0 to 63
    """
    def imm_forms(self):
        return [[minmax_constraint(minval=0, maxval=63)]]

    def with_register(self, c, a, b):
        return self.asmwrap(f"lsl {c},{a},{b}")

    def with_immediate(self, c, a, imm):
        return self.asmwrap(f"lsl {c},{a},#{imm}")


class aarch64_shr(scalar_arith):
    """
    lsr: logical shift right, by 0 to 63
    """
    def imm_forms(self):
        return [[minmax_constraint(minval=0, maxval=63)]]

    def with_register(self, c, a, b):
        return self.asmwrap(f"lsr {c},{a},{b}")

    def with_immediate(self, c, a, imm):
        return self.asmwrap(f"lsr {c},{a},#{imm}")

