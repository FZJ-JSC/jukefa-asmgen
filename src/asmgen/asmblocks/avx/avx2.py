# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
X86_64 AVX/AVX2+FMA generator specifics (xmm/ymm, no masking)
"""


from ..x86.opmem import x86_prefetch
from ..x86.optransform import x86_shift, x86_sub
from .base import avxbase
from .opmem.load import avx128_load, avx256_load
from .opmem.store import avx128_store, avx256_store
from .opmove import avx_move
from .opset import avx_opset
from .optransform import avx_add, avx_fma, avx_mul
from .types import xmm_vreg, ymm_vreg


class fma128(avxbase):
    """
    X86_64/AVX/FMA 128 bit asmgem implementation
    """

    def __init__(self):
        super().__init__()
        self.fma = avx_fma(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     rpref=self.rpref,
                     has_fp16=False
                     )
        self.mul = avx_mul(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     rpref=self.rpref,
                     has_fp16=False
                     )
        self.add = avx_add(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     rpref=self.rpref,
                     has_fp16=False
                     )

        self.load = avx128_load(asmwrap=self.asmwrap, rpref=self.rpref)
        self.store = avx128_store(asmwrap=self.asmwrap, rpref=self.rpref)
        self.prefetch = x86_prefetch(asmwrap=self.asmwrap, rpref=self.rpref)
        self.move = avx_move(asmwrap=self.asmwrap, rpref=self.rpref, simd_bytes=16)
        self.sub = x86_sub(asmwrap=self.asmwrap, rpref=self.rpref)
        self.shift = x86_shift(asmwrap=self.asmwrap, rpref=self.rpref)
        self.set = avx_opset(asmwrap=self.asmwrap, rpref=self.rpref, evex=False, simd_bytes=self.simd_size,
                              index_table=self.index_table)
        # the operations exist: the features that are off hide their signatures
        self._gate_features()

    def get_req_flags(self):
        return ['fma', 'avx']
    def max_fregs(self):
        return 16
    def max_vregs(self):
        return 16

    @property
    def simd_size(self):
        return 16

    def vreg(self, reg_idx):
        return xmm_vreg(reg_idx)

class fma256(avxbase):
    """
    X86_64/AVX/FMA 256 bit asmgem implementation
    """

    def __init__(self):
        super().__init__()
        self.fma = avx_fma(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     rpref=self.rpref,
                     has_fp16=False
                     )
        self.mul = avx_mul(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     rpref=self.rpref,
                     has_fp16=False
                     )
        self.add = avx_add(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     rpref=self.rpref,
                     has_fp16=False
                     )

        self.load = avx256_load(asmwrap=self.asmwrap, rpref=self.rpref)
        self.store = avx256_store(asmwrap=self.asmwrap, rpref=self.rpref)
        self.prefetch = x86_prefetch(asmwrap=self.asmwrap, rpref=self.rpref)
        self.move = avx_move(asmwrap=self.asmwrap, rpref=self.rpref, simd_bytes=32)
        self.sub = x86_sub(asmwrap=self.asmwrap, rpref=self.rpref)
        self.shift = x86_shift(asmwrap=self.asmwrap, rpref=self.rpref)
        self.set = avx_opset(asmwrap=self.asmwrap, rpref=self.rpref, evex=False, simd_bytes=self.simd_size,
                              index_table=self.index_table)
        # the operations exist: the features that are off hide their signatures
        self._gate_features()

    def get_req_flags(self):
        return ['fma', 'avx']
    def max_fregs(self):
        return 16
    def max_vregs(self):
        return 16

    @property
    def simd_size(self):
        return 32

    def vreg(self, reg_idx : int):
        return ymm_vreg(reg_idx)
