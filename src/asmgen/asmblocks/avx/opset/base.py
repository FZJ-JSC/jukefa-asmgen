"""
avx's set
"""
from typing import Callable

from ...op.opset import vector_opset
from ....registers import asm_data_type as adt, data_reg
from ...x86.opset import x86_opset
from .signatures import make_avx_opset_signatures


class avx_opset(vector_opset):
    """
    AVX set: vpxor v,v,v - vpxorq with AVX-512 (evex); general and
    floating-point registers x86's
    """
    def __init__(self, *, asmwrap : Callable[[str], str], rpref : Callable, evex : bool = False):
        self.asmwrap = asmwrap
        self.rpref = rpref
        self.evex = evex
        self.scalar_op = x86_opset(asmwrap=asmwrap, rpref=rpref, evex=evex)
        self.signatures = make_avx_opset_signatures()

    def zero_reg(self, *, dreg : data_reg, dt : adt) -> str:
        p = self.rpref(dreg)
        return self.asmwrap(f"{'vpxorq' if self.evex else 'vpxor'} {p},{p},{p}")
