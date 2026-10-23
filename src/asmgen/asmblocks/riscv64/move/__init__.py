"""
RISC-V moves between scalar registers
"""
from ....registers import asm_data_type as adt
from ...op.scalar import scalar_move


class riscv64_move(scalar_move):
    """
    add rd,rs,0 (general), fmv.{d,s,h} (floating-point), fmv.{d,w,h}.x (bits)
    """
    FCDT = {adt.DOUBLE: "d", adt.SINGLE: "s", adt.HALF: "h"}
    FDT = {adt.DOUBLE: "d", adt.SINGLE: "w", adt.HALF: "h"}

    def greg_to_greg(self, src, dst):
        return self.asmwrap(f"add {dst},{src},0")

    def freg_to_freg(self, src, dst, dt):
        return self.asmwrap(f"fmv.{self.FCDT[dt]} {dst},{src}")

    def bits_to_freg(self, src, dst, dt):
        return self.asmwrap(f"fmv.{self.FDT[dt]}.x {dst},{src}")
