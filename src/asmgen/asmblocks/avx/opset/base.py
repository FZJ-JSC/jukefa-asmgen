"""
avx's set
"""
from typing import Callable

from ...op.opset import vector_opset
from ....registers import asm_data_type as adt, adt_size, data_reg, greg_base
from ..types import avx512_mreg
from ...x86.opset import x86_opset
from .signatures import make_avx_opset_signatures
from ...x86.opset.base import ONE_SHIFTS
from ...x86.types import SIZE_SUFFIXES, x86_greg

# 1 from all bits set: a logical right shift by the element's bits less one
INT_SHIFTS = {adt.UINT64: ("q", 63), adt.SINT64: ("q", 63), adt.UINT32: ("d", 31), adt.SINT32: ("d", 31),
              adt.UINT16: ("w", 15), adt.SINT16: ("w", 15)}


class avx_opset(vector_opset):
    """
    AVX set: vpxor v,v,v - vpxorq with AVX-512 (evex); general and
    floating-point registers x86's
    """
    def __init__(self, *, asmwrap : Callable[[str], str], rpref : Callable, evex : bool = False,
                 simd_bytes : int = 32, index_table : Callable[[adt], str] | None = None):
        self.asmwrap = asmwrap
        self.index_table = index_table
        self.simd_bytes = simd_bytes
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

    def maskavl_reg(self, *, dreg : avx512_mreg, dt : adt, avlreg : x86_greg,
                    tmpreg : x86_greg | None = None) -> str:
        """
        AVX-512: the first avlreg lanes true - all ones in a scratch general
        register, bzhi keeping the low avlreg bits (BMI2), kmovw into the k
        register: 16 lanes or fewer
        """
        if not isinstance(dreg, avx512_mreg):
            raise ValueError(f"a mask of a count is a k register, not {dreg}")
        if tmpreg is None:
            raise ValueError("AVX-512's mask of a count takes a scratch general register (tmpreg)")
        if tmpreg.idx == avlreg.idx:
            raise ValueError(f"the scratch register {tmpreg} is the count's")
        t = f"%{tmpreg.name(8)}"
        return (self.asmwrap(f"movq $-1,{t}") + self.asmwrap(f"bzhiq %{avlreg.name(8)},{t},{t}")
                + self.asmwrap(f"kmovw %{tmpreg.name(4)},{self.rpref(dreg)}"))

    def mask_reg(self, *, dreg : data_reg, dt : adt, mask : str, count : int | None = None) -> str:
        p = self.rpref(dreg)
        if "first" == mask:
            # every bit set, then shifted right: the low count bits stay
            return self.asmwrap(f"kxnorw {p},{p},{p}") + self.asmwrap(f"kshiftrw ${16 - count},{p},{p}")
        if isinstance(dreg, avx512_mreg):
            return self.asmwrap(f"{'kxnorw' if 'alltrue' == mask else 'kxorw'} {p},{p},{p}")
        # a vector register as a mask (AVX2): every bit set, or none
        return self.asmwrap(f"{'vpcmpeqd' if 'alltrue' == mask else 'vpxor'} {p},{p},{p}")

    def vlen(self, *, dreg : greg_base, dt : adt = adt.UINT8) -> str:
        return self.scalar_op.value_greg(dreg=dreg, value=self.simd_bytes // adt_size(dt))

    def stindex_reg(self, *, dreg : data_reg, dt : adt, streg : greg_base) -> str:
        # the stride broadcast, times 0, 1, 2, ... - a table the generator
        # keeps in its data (index_table: registered, its label as written)
        size = adt_size(dt)
        suf = SIZE_SUFFIXES[size]
        label = self.index_table(dt)
        # the broadcast's source: r32 for 1 and 2 bytes too
        pst, pv = self.rpref(streg, size=max(4, size)), self.rpref(dreg)
        result = self.asmwrap(f"vpbroadcast{suf} {pst},{pv}")
        if 8 == size:
            # vpmullq is AVX512DQ; for strides below 4 GB the low dwords
            # suffice (vpmuludq, AVX512F on zmm)
            return result + self.asmwrap(f"vpmuludq {label}({self.rpref('rip')}),{pv},{pv}")
        return result + self.asmwrap(f"vpmull{suf} {label}({self.rpref('rip')}),{pv},{pv}")
