"""
SVE's add on general registers: AArch64's, and counts of vector lengths
"""
from ...aarch64.opd3 import aarch64_add
from ...op.constraint import minmax_constraint


class sve_scalar_add(aarch64_add):
    """
    AArch64's add, and imm{VLSCALE}: addvl's -32..31 vector lengths - incb
    or decb (1..16) where the destination is the source
    """
    def vl_forms(self):
        return [[minmax_constraint(minval=-32, maxval=31)]]

    def with_vl(self, c, a, count):
        if str(c) == str(a) and 1 <= abs(count) <= 16:
            return self.asmwrap(f"{'incb' if count > 0 else 'decb'} {c}, ALL, MUL #{abs(count)}")
        return self.asmwrap(f"addvl {c},{a},#{count}")
