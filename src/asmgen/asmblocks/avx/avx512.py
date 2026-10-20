# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
X86_64 AVX512 generator specifics (zmm, k mask regs, masking)
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
from .opd3 import avx_fma,avx_fmul,avx_fadd

from .opmem.load import avx128_load,avx256_load,avx512_load
from .opmem.store import avx128_store,avx256_store,avx512_store

class avx512(avxbase):
    """
    X86_64/AVX512 asmgem implementation
    """
    # AVX512VL, AVX512DQ, AVX512BW, AVX512-FP16, AVX512-BF16
    FEATURES = ('VL', 'DQ', 'BW', 'FP16', 'BF16')


    def __init__(self):
        super().__init__()
        self.fma = avx_fma(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     rpref=self.rpref,
                     has_fp16=True
                     )
        self.fmul = avx_fmul(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     rpref=self.rpref,
                     has_fp16=True
                     )
        self.fadd = avx_fadd(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     rpref=self.rpref,
                     has_fp16=True
                     )

        self.load = avx512_load(asmwrap=self.asmwrap, rpref=self.rpref)
        self.store = avx512_store(asmwrap=self.asmwrap, rpref=self.rpref)
        self.move = avx_move(asmwrap=self.asmwrap, rpref=self.rpref, simd_bytes=64)

    def get_req_flags(self) -> list[str]:
        return ['avx512f']
    def max_fregs(self):
        return 32
    def max_vregs(self):
        return 32
    def max_mregs(self):
        return 8

    @property
    def simd_size(self):
        return 64



    def zero_vreg(self, *, vreg : vreg_base, dt : adt):
        preg = self.rpref(vreg)
        return self.asmwrap(f"vpxorq {preg},{preg},{preg}")

    def zero_freg(self, *, freg : freg_base, dt : adt) -> str:
        # EVEX: FP scalars live in the vector file, allocated from the top -
        # xmm16-31 VEX's vpxor can't reach (AVX512VL)
        preg = self.rpref(freg)
        return self.asmwrap(f"vpxord {preg},{preg},{preg}")

    def vreg(self, reg_idx : int):
        return zmm_vreg(reg_idx)

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

    def load_vector_gather(self, *, areg : greg_base, offvreg : vreg_base,
                           vreg : vreg_base, dt : adt,
                           it : ait):
        suf = 'p'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pv = self.rpref(vreg)
        pov = self.rpref(offvreg)
        address = f"({pa},{pov},1)"
        isuf = self.it_suffixes[it]
        maskreg = '%k2'
        if self.output_inline:
            maskreg = '%%k2'
        masksuf = self.size_mask_suffixes[adt_size(dt)]

        asmblock = self.asmwrap(f"kxnor{masksuf} {maskreg},{maskreg},{maskreg}")
        asmblock += self.asmwrap(f"vgather{isuf}{suf} {address},{pv}{{{maskreg}}}")
        return asmblock

    def store_vector_scatter(self, *, areg : greg_base, offvreg : vreg_base,
                           vreg : vreg_base, dt : adt,
                           it : ait):
        suf = 'p'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pv = self.rpref(vreg)
        pov = self.rpref(offvreg)
        address = f"({pa},{pov},1)"
        isuf = self.it_suffixes[it]
        maskreg = '%k2'
        if self.output_inline:
            maskreg = '%%k2'
        masksuf = self.size_mask_suffixes[adt_size(dt)]

        asmblock = self.asmwrap(f"kxnor{masksuf} {maskreg},{maskreg},{maskreg}")
        asmblock += self.asmwrap(f"vscatter{isuf}{suf} {pv},{address}{{{maskreg}}}")
        return asmblock
