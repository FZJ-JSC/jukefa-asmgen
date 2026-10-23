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
