"""
RISC-V scalar arithmetic on general registers: add, sub, mul, shift
"""
from ...op.constraint import minmax_constraint
from ...op.optransform import optransform_modifier
from ...op.scalar import scalar_arith
from ....registers import greg_base


class riscv64_add(scalar_arith):
    """
    add; addi's 12-bit signed immediate; Zba's sh1add..sh3add (ZBA)
    """
    def imm_forms(self):
        return [[minmax_constraint(minval=-2048, maxval=2047)]]

    def shift_forms(self):
        return [([minmax_constraint(minval=1, maxval=3)], frozenset({"ZBA"}))]

    def with_register(self, *, adreg : greg_base, bdreg : greg_base, cdreg : greg_base, modifiers : set) -> str:
        return self.asmwrap(f"add {cdreg},{adreg},{bdreg}")

    def with_immediate(self, *, adreg : greg_base, bimm : int, cdreg : greg_base, modifiers : set) -> str:
        return self.asmwrap(f"add {cdreg},{adreg},{bimm}")

    def with_shifted(self, *, adreg : greg_base, bdreg : greg_base, bdreg_shift : int, cdreg : greg_base) -> str:
        # shNadd rd,rs1,rs2: rd = (rs1 << N) + rs2
        return self.asmwrap(f"sh{bdreg_shift}add {cdreg},{bdreg},{adreg}")


class riscv64_sub(scalar_arith):
    """
    sub; an immediate is addi's, negated
    """
    def imm_forms(self):
        return [[minmax_constraint(minval=-2047, maxval=2048)]]

    def with_register(self, *, adreg : greg_base, bdreg : greg_base, cdreg : greg_base, modifiers : set) -> str:
        return self.asmwrap(f"sub {cdreg},{adreg},{bdreg}")

    def with_immediate(self, *, adreg : greg_base, bimm : int, cdreg : greg_base, modifiers : set) -> str:
        return self.asmwrap(f"add {cdreg},{adreg},{-bimm}")


class riscv64_mul(scalar_arith):
    """
    mul (M); no immediate form
    """
    def with_register(self, *, adreg : greg_base, bdreg : greg_base, cdreg : greg_base, modifiers : set) -> str:
        return self.asmwrap(f"mul {cdreg},{adreg},{bdreg}")


class riscv64_shift(scalar_arith):
    """
    sll, srl, slli, srli (by 0 to 63): optransform_modifier SHL or SHR
    """
    def op_modifiers(self):
        return [{optransform_modifier.SHL}, {optransform_modifier.SHR}]

    def imm_forms(self):
        return [[minmax_constraint(minval=0, maxval=63)]]

    def with_register(self, *, adreg : greg_base, bdreg : greg_base, cdreg : greg_base, modifiers : set) -> str:
        return self.asmwrap(f"{'sll' if optransform_modifier.SHL in modifiers else 'srl'} {cdreg},{adreg},{bdreg}")

    def with_immediate(self, *, adreg : greg_base, bimm : int, cdreg : greg_base, modifiers : set) -> str:
        return self.asmwrap(f"{'slli' if optransform_modifier.SHL in modifiers else 'srli'} {cdreg},{adreg},{bimm}")
