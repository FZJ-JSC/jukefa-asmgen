"""
x86's set
"""
from typing import Callable

from ...op.opset import scalar_opset
from ....registers import asm_data_type as adt, greg_base, freg_base
from .signatures import make_x86_opset_signatures


class x86_opset(scalar_opset):
    """
    x86-64 set: movq $0; vpxor - vpxord with AVX-512 (evex)
    """
    def __init__(self, *, asmwrap : Callable[[str], str], rpref : Callable, evex : bool = False):
        self.rpref = rpref
        self.evex = evex
        super().__init__(asmwrap=asmwrap)

    def make_signatures(self):
        return make_x86_opset_signatures()

    def zero_greg(self, *, dreg : greg_base) -> str:
        return self.asmwrap(f"movq $0,{self.rpref(dreg)}")

    def zero_freg(self, *, dreg : freg_base, dt : adt) -> str:
        p = self.rpref(dreg)
        return self.asmwrap(f"{'vpxord' if self.evex else 'vpxor'} {p},{p},{p}")
