"""
RISC-V scalar arithmetic on general registers: add, sub, mul
"""
from ...op.constraint import minmax_constraint
from ...op.scalar import scalar_arith


class riscv64_add(scalar_arith):
    """
    add; addi's 12-bit signed immediate
    """
    def imm_forms(self):
        return [[minmax_constraint(minval=-2048, maxval=2047)]]

    def with_register(self, c, a, b):
        return self.asmwrap(f"add {c},{a},{b}")

    def with_immediate(self, c, a, imm):
        return self.asmwrap(f"add {c},{a},{imm}")

    def shift_forms(self):
        # Zba's sh1add, sh2add, sh3add: rd = (rs1 << n) + rs2
        return [([minmax_constraint(minval=1, maxval=3)], frozenset({"ZBA"}))]

    def with_shifted(self, c, a, b, shift):
        return self.asmwrap(f"sh{shift}add {c},{b},{a}")


class riscv64_sub(scalar_arith):
    """
    sub; an immediate is addi's, negated
    """
    def imm_forms(self):
        return [[minmax_constraint(minval=-2047, maxval=2048)]]

    def with_register(self, c, a, b):
        return self.asmwrap(f"sub {c},{a},{b}")

    def with_immediate(self, c, a, imm):
        return self.asmwrap(f"add {c},{a},{-imm}")


class riscv64_mul(scalar_arith):
    """
    mul (M); no immediate form
    """
    def with_register(self, c, a, b):
        return self.asmwrap(f"mul {c},{a},{b}")


class riscv64_shl(scalar_arith):
    """
    sll, slli: logical shift left, by 0 to 63
    """
    def imm_forms(self):
        return [[minmax_constraint(minval=0, maxval=63)]]

    def with_register(self, c, a, b):
        return self.asmwrap(f"sll {c},{a},{b}")

    def with_immediate(self, c, a, imm):
        return self.asmwrap(f"slli {c},{a},{imm}")


class riscv64_shr(scalar_arith):
    """
    srl, srli: logical shift right, by 0 to 63
    """
    def imm_forms(self):
        return [[minmax_constraint(minval=0, maxval=63)]]

    def with_register(self, c, a, b):
        return self.asmwrap(f"srl {c},{a},{b}")

    def with_immediate(self, c, a, imm):
        return self.asmwrap(f"srli {c},{a},{imm}")

