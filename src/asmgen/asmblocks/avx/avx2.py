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

from .move import avx_move
from .opd3 import avx_fma,avx_mul,avx_add

from .opmem.load import avx128_load,avx256_load,avx512_load
from .opmem.store import avx128_store,avx256_store,avx512_store
from ..x86.opd3 import x86_sub
from ..x86.opd3 import x86_shl, x86_shr
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
        self.move = avx_move(asmwrap=self.asmwrap, rpref=self.rpref, simd_bytes=16)
        self.sub = x86_sub(asmwrap=self.asmwrap, rpref=self.rpref)
        self.shl = x86_shl(asmwrap=self.asmwrap, rpref=self.rpref)
        self.shr = x86_shr(asmwrap=self.asmwrap, rpref=self.rpref)
        self.set = avx_opset(asmwrap=self.asmwrap, rpref=self.rpref, evex=False)

    def get_req_flags(self):
        return ['fma', 'avx']
    def max_fregs(self):
        return 16
    def max_vregs(self):
        return 16

    @property
    def simd_size(self):
        return 16

    def zero_vreg(self, *, vreg : vreg_base, dt : adt):
        return self.set(dreg=vreg, dt=dt, modifiers={_opset_mod.CONST}, constant="zero")

    def vreg(self, reg_idx):
        return xmm_vreg(reg_idx)

    def load_vector_bcast1(self, *, areg : greg_base,
                          vreg : vreg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pv = self.rpref(self.xmm_to_ymm(vreg))
        return self.asmwrap(f"vbroadcast{suf} ({pa}),{pv}")

    def load_vector_bcast1_immoff(self, *, areg : greg_base, offset : int,
                               vreg : vreg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pv = self.rpref(self.xmm_to_ymm(vreg))
        return self.asmwrap(f"vbroadcast{suf} {offset}({pa}),{pv}")

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
        self.move = avx_move(asmwrap=self.asmwrap, rpref=self.rpref, simd_bytes=32)
        self.sub = x86_sub(asmwrap=self.asmwrap, rpref=self.rpref)
        self.shl = x86_shl(asmwrap=self.asmwrap, rpref=self.rpref)
        self.shr = x86_shr(asmwrap=self.asmwrap, rpref=self.rpref)
        self.set = avx_opset(asmwrap=self.asmwrap, rpref=self.rpref, evex=False)

    def get_req_flags(self):
        return ['fma', 'avx']
    def max_fregs(self):
        return 16
    def max_vregs(self):
        return 16

    @property
    def simd_size(self):
        return 32

    def zero_vreg(self, *, vreg : vreg_base, dt : adt):
        return self.set(dreg=vreg, dt=dt, modifiers={_opset_mod.CONST}, constant="zero")

    def vreg(self, reg_idx : int):
        return ymm_vreg(reg_idx)

    def load_vector_bcast1(self, *, areg : greg_base,
                          vreg : vreg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pv = self.rpref(vreg)
        return self.asmwrap(f"vbroadcast{suf} ({pa}),{pv}")

    def load_vector_bcast1_immoff(self, *, areg : greg_base, offset : int,
                               vreg : vreg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pv = self.rpref(vreg)
        return self.asmwrap(f"vbroadcast{suf} {offset}({pa}),{pv}")

