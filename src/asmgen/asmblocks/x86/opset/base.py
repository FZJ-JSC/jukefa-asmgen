"""
x86's set
"""
from typing import Callable

from ...op.opset import scalar_opset
from ....registers import asm_data_type as adt, greg_base, freg_base
from .signatures import make_x86_opset_signatures

# 1.0 from all bits set: shifted left, then right logically by 2
ONE_SHIFTS = {adt.DOUBLE: ("q", 54), adt.SINGLE: ("d", 25), adt.HALF: ("w", 12)}


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

    def all_bits(self, *, p : str) -> str:
        """
        Every bit of the named register set: vpcmpeqd, vpternlogd with
        AVX-512 (evex: any of 32 registers)
        """
        if self.evex:
            return self.asmwrap(f"vpternlogd $0xff,{p},{p},{p}")
        return self.asmwrap(f"vpcmpeqd {p},{p},{p}")

    def one_freg(self, *, dreg : freg_base, dt : adt) -> str:
        p = self.rpref(dreg)
        suf, left = ONE_SHIFTS[dt]
        return (self.all_bits(p=p) + self.asmwrap(f"vpsll{suf} ${left},{p},{p}")
                + self.asmwrap(f"vpsrl{suf} $2,{p},{p}"))

    def value_greg(self, *, dreg : greg_base, value : int) -> str:
        # movq takes any 64-bit value: the assembler encodes those past 32
        # bits as movabs; past 2**63, the same bits signed
        return self.asmwrap(f"movq ${value - 2**64 if value >= 2**63 else value},{self.rpref(dreg)}")
