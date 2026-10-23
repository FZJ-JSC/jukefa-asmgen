"""
AArch64 moves between scalar registers
"""
from ....registers import adt_size
from ...op.scalar import scalar_move


class aarch64_move(scalar_move):
    """
    mov (general), fmov (floating-point, and bits: from x, or w for 4 and 2
    bytes)
    """
    def greg_to_greg(self, src, dst):
        return self.asmwrap(f"mov {dst},{src}")

    def freg_to_freg(self, src, dst, dt):
        return self.asmwrap(f"fmov {dst},{src}")

    def bits_to_freg(self, src, dst, dt):
        if 8 == adt_size(dt):
            return self.asmwrap(f"fmov {dst},{src}")
        return self.asmwrap(f"fmov {dst},w{str(src)[1:]}")
