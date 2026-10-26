"""
AArch64 moves between scalar registers
"""
from ...op.scalar import scalar_move
from ....registers import asm_data_type as adt, adt_size, greg_base, freg_base


class aarch64_move(scalar_move):
    """
    mov (general), fmov (floating-point, and bits: from x, or w for 4 and 2
    bytes)
    """
    def greg_to_greg(self, *, src : greg_base, dst : greg_base) -> str:
        return self.asmwrap(f"mov {dst},{src}")

    def freg_to_freg(self, *, src : freg_base, dst : freg_base, dt : adt) -> str:
        return self.asmwrap(f"fmov {dst},{src}")

    def bits_to_freg(self, *, src : greg_base, dst : freg_base, dt : adt) -> str:
        if 8 == adt_size(dt):
            return self.asmwrap(f"fmov {dst},{src}")
        return self.asmwrap(f"fmov {dst},w{str(src)[1:]}")
