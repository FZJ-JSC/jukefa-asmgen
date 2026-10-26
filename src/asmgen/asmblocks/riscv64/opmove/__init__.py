"""
RISC-V moves between scalar registers
"""
from ...op.scalar import scalar_move
from ....registers import asm_data_type as adt, greg_base, freg_base
from ..types import FCDT_SUFFIXES, FDT_SUFFIXES


class riscv64_move(scalar_move):
    """
    add rd,rs,0 (general), fmv.{d,s,h} (floating-point), fmv.{d,w,h}.x (bits)
    """
    def greg_to_greg(self, *, src : greg_base, dst : greg_base) -> str:
        return self.asmwrap(f"add {dst},{src},0")

    def freg_to_freg(self, *, src : freg_base, dst : freg_base, dt : adt) -> str:
        return self.asmwrap(f"fmv.{FCDT_SUFFIXES[dt]} {dst},{src}")

    def bits_to_freg(self, *, src : greg_base, dst : freg_base, dt : adt) -> str:
        return self.asmwrap(f"fmv.{FDT_SUFFIXES[dt]}.x {dst},{src}")
