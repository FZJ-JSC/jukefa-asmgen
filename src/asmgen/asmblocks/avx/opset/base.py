"""
avx's set
"""
from typing import Callable

from ...op.opset import vector_opset
from ....registers import asm_data_type as adt, data_reg
from ..types import avx512_mreg
from ...x86.opset import x86_opset
from .signatures import make_avx_opset_signatures
from ...x86.opset.base import ONE_SHIFTS

# 1 from all bits set: a logical right shift by the element's bits less one
INT_SHIFTS = {adt.UINT64: ("q", 63), adt.SINT64: ("q", 63), adt.UINT32: ("d", 31), adt.SINT32: ("d", 31),
              adt.UINT16: ("w", 15), adt.SINT16: ("w", 15)}


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
        self.signatures = make_avx_opset_signatures(evex=evex)

    def zero_reg(self, *, dreg : data_reg, dt : adt) -> str:
        p = self.rpref(dreg)
        return self.asmwrap(f"{'vpxorq' if self.evex else 'vpxor'} {p},{p},{p}")

    def one_reg(self, *, dreg : data_reg, dt : adt) -> str:
        p = self.rpref(dreg)
        ones = self.scalar_op.all_bits(p=p)
        if dt in ONE_SHIFTS:
            suf, left = ONE_SHIFTS[dt]
            return ones + self.asmwrap(f"vpsll{suf} ${left},{p},{p}") + self.asmwrap(f"vpsrl{suf} $2,{p},{p}")
        if dt in INT_SHIFTS:
            suf, right = INT_SHIFTS[dt]
            return ones + self.asmwrap(f"vpsrl{suf} ${right},{p},{p}")
        # bytes: |-1| = 1
        return ones + self.asmwrap(f"vpabsb {p},{p}")

    def mask_reg(self, *, dreg : data_reg, dt : adt, mask : str) -> str:
        p = self.rpref(dreg)
        if isinstance(dreg, avx512_mreg):
            return self.asmwrap(f"{'kxnorw' if 'alltrue' == mask else 'kxorw'} {p},{p},{p}")
        # a vector register as a mask (AVX2): every bit set, or none
        return self.asmwrap(f"{'vpcmpeqd' if 'alltrue' == mask else 'vpxor'} {p},{p},{p}")
