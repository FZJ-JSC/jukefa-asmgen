"""
SVE's add on general registers: AArch64's, and counts of vector lengths
"""
from ...aarch64.optransform import aarch64_add
from ...op.constraint import minmax_constraint
from ....registers import greg_base


class sve_scalar_add(aarch64_add):
    """
    AArch64's add, and VOFFSET: bimm counts vector lengths, addvl's -32..31
    - incb or decb (1..16) where the destination is the source
    """
    def voffset_forms(self):
        return [[minmax_constraint(minval=-32, maxval=31)]]

    def with_voffset(self, *, adreg : greg_base, bimm : int, cdreg : greg_base) -> str:
        if str(cdreg) == str(adreg) and 1 <= abs(bimm) <= 16:
            return self.asmwrap(f"{'incb' if bimm > 0 else 'decb'} {cdreg}, ALL, MUL #{abs(bimm)}")
        return self.asmwrap(f"addvl {cdreg},{adreg},#{bimm}")
