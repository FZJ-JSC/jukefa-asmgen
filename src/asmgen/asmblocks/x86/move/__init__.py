"""
x86-64 moves between scalar registers
"""
from typing import Callable

from ....registers import asm_data_type as adt, adt_size
from ...op.scalar import scalar_move


class x86_move(scalar_move):
    """
    movq (general), vmovs{d,s,h} (floating-point), vmovq/vmovd (bits)
    """
    SUFFIX = {adt.DOUBLE: "d", adt.SINGLE: "s", adt.HALF: "h"}

    def __init__(self, *, asmwrap : Callable[[str], str], rpref : Callable):
        self.rpref = rpref
        super().__init__(asmwrap=asmwrap)

    def greg_to_greg(self, src, dst):
        return self.asmwrap(f"movq {self.rpref(src)}, {self.rpref(dst)}")

    def freg_to_freg(self, src, dst, dt):
        return self.asmwrap(f"vmovs{self.SUFFIX[dt]} {src}, {dst}")

    def bits_to_freg(self, src, dst, dt):
        if 8 == adt_size(dt):
            return self.asmwrap(f"vmovq {self.rpref(src)},{self.rpref(dst)}")
        return self.asmwrap(f"vmovd {self.rpref(src, size=4)},{self.rpref(dst)}")
