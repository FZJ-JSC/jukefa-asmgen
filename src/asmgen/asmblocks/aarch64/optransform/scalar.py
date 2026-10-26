"""
AArch64 scalar arithmetic on general registers: add, sub, mul, shift
"""
from ...op.constraint import anyof_constraint, minmax_constraint, multiple_constraint
from ...op.optransform import optransform_modifier
from ...op.scalar import scalar_arith
from ....registers import greg_base

SHIFTED_MAX = 4095 * 4096


def _imm_forms():
    # 12 bits, or 12 bits shifted left by 12; a negative value is the other
    # operation's (the assembler turns add #-8 into sub #8)
    return [[anyof_constraint(alternatives=[
        [minmax_constraint(minval=-4095, maxval=4095)],
        [multiple_constraint(multiple=4096), minmax_constraint(minval=-SHIFTED_MAX, maxval=SHIFTED_MAX)],
    ])]]


class aarch64_addsub(scalar_arith):
    """
    add or sub (mnemonic): immediates, and B shifted left by 0 to 63
    """
    mnemonic = None

    def imm_forms(self):
        return _imm_forms()

    def shift_forms(self):
        return [([minmax_constraint(minval=0, maxval=63)], frozenset())]

    def with_register(self, *, adreg : greg_base, bdreg : greg_base, cdreg : greg_base, modifiers : set) -> str:
        return self.asmwrap(f"{self.mnemonic} {cdreg},{adreg},{bdreg}")

    def with_immediate(self, *, adreg : greg_base, bimm : int, cdreg : greg_base, modifiers : set) -> str:
        return self.asmwrap(f"{self.mnemonic} {cdreg},{adreg},#{bimm}")

    def with_shifted(self, *, adreg : greg_base, bdreg : greg_base, bdreg_shift : int, cdreg : greg_base) -> str:
        if not bdreg_shift:
            return self.asmwrap(f"{self.mnemonic} {cdreg},{adreg},{bdreg}")
        return self.asmwrap(f"{self.mnemonic} {cdreg},{adreg},{bdreg}, LSL #{bdreg_shift}")


class aarch64_add(aarch64_addsub):
    """
    add
    """
    mnemonic = "add"


class aarch64_sub(aarch64_addsub):
    """
    sub
    """
    mnemonic = "sub"


class aarch64_mul(scalar_arith):
    """
    mul; no immediate form
    """
    def with_register(self, *, adreg : greg_base, bdreg : greg_base, cdreg : greg_base, modifiers : set) -> str:
        return self.asmwrap(f"mul {cdreg},{adreg},{bdreg}")


class aarch64_shift(scalar_arith):
    """
    lsl, lsr (by 0 to 63): optransform_modifier SHL or SHR
    """
    def op_modifiers(self):
        return [{optransform_modifier.SHL}, {optransform_modifier.SHR}]

    def imm_forms(self):
        return [[minmax_constraint(minval=0, maxval=63)]]

    def with_register(self, *, adreg : greg_base, bdreg : greg_base, cdreg : greg_base, modifiers : set) -> str:
        return self.asmwrap(f"{'lsl' if optransform_modifier.SHL in modifiers else 'lsr'} {cdreg},{adreg},{bdreg}")

    def with_immediate(self, *, adreg : greg_base, bimm : int, cdreg : greg_base, modifiers : set) -> str:
        return self.asmwrap(f"{'lsl' if optransform_modifier.SHL in modifiers else 'lsr'} {cdreg},{adreg},#{bimm}")
