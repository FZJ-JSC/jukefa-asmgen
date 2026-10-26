# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
X86_64 AVX/AVX2+FMA generator specifics (xmm/ymm, no masking)
"""

from copy import deepcopy
from typing import Union
from abc import abstractmethod

from ..x86.opmem import x86_prefetch
from ...asmdata import asm_data
from ...registers import (
    reg_tracker,
    asm_data_type as adt,
    adt_size,
    asm_index_type as ait,
    treg_base, vreg_base, freg_base, greg_base
)

from ..noarch import asmgen,comparison
from ...callconv.callconv import callconv

from .base import avxbase
from .types import x86_greg,avx_freg,xmm_vreg,ymm_vreg,zmm_vreg,reg_prefixer

from .opmove import avx_move
from .optransform import avx_fma,avx_mul,avx_add

from .opmem.load import avx128_load,avx256_load,avx512_load
from .opmem.store import avx128_store,avx256_store,avx512_store
from ..x86.optransform import x86_sub
from ..x86.optransform import x86_shift
from .opset import avx_opset
from ..op.opset import opset_modifier as _opset_mod

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
