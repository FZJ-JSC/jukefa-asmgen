"""
x86-64 moves between scalar registers
"""
from typing import Callable

from ...op.scalar import scalar_move
from ....registers import asm_data_type as adt, adt_size, greg_base, freg_base
from ..types import DT_SUFFIXES


class x86_move(scalar_move):
    """
    movq (general), vmovs{d,s,h} (floating-point), vmovq/vmovd (bits)
    """
    def __init__(self, *, asmwrap : Callable[[str], str], rpref : Callable):
        self.rpref = rpref
        super().__init__(asmwrap=asmwrap)

    def greg_to_greg(self, *, src : greg_base, dst : greg_base) -> str:
        return self.asmwrap(f"movq {self.rpref(src)}, {self.rpref(dst)}")

    def freg_to_freg(self, *, src : freg_base, dst : freg_base, dt : adt) -> str:
        return self.asmwrap(f"vmovs{DT_SUFFIXES[dt]} {src}, {dst}")

    def bits_to_freg(self, *, src : greg_base, dst : freg_base, dt : adt) -> str:
        if 8 == adt_size(dt):
            return self.asmwrap(f"vmovq {self.rpref(src)},{self.rpref(dst)}")
        return self.asmwrap(f"vmovd {self.rpref(src, size=4)},{self.rpref(dst)}")
