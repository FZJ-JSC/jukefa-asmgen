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

from ..noarch import compiler_family
from ..x86.opmem import x86_prefetch
from ...asmdata import asm_data
from ...registers import asm_index_type
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
from ..x86.opd3 import x86_shift
from .opset import avx_opset
from ..op.opset import opset_modifier as _opset_mod
from .types import avx512_mreg

class avx512(avxbase):
    """
    X86_64/AVX512 asmgem implementation
    """
    # AVX512VL, AVX512DQ, AVX512BW, AVX512-FP16, AVX512-BF16
    FEATURES = ('VL', 'DQ', 'BW', 'FP16', 'BF16')
    #: the features as compiler flags
    FEATURE_FLAGS = {"VL": "-mavx512vl", "DQ": "-mavx512dq", "BW": "-mavx512bw",
                     "FP16": "-mavx512fp16", "BF16": "-mavx512bf16"}

    def compiler_flags(self, compiler : str = "gcc") -> list[str]:
        if "arm" == compiler_family(compiler):
            raise NotImplementedError("armclang targets no x86")
        enabled = self.enabled_features()
        return ["-mavx512f"] + [self.FEATURE_FLAGS[f] for f in self.FEATURES if f in enabled]


    def __init__(self):
        super().__init__()
        self.fma = avx_fma(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     rpref=self.rpref,
                     has_fp16=True
                     )
        self.mul = avx_mul(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     rpref=self.rpref,
                     has_fp16=True
                     )
        self.add = avx_add(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     rpref=self.rpref,
                     has_fp16=True
                     )

        self.load = avx512_load(asmwrap=self.asmwrap, rpref=self.rpref)
        self.store = avx512_store(asmwrap=self.asmwrap, rpref=self.rpref)
        self.prefetch = x86_prefetch(asmwrap=self.asmwrap, rpref=self.rpref)
        self.move = avx_move(asmwrap=self.asmwrap, rpref=self.rpref, simd_bytes=64)
        self.sub = x86_sub(asmwrap=self.asmwrap, rpref=self.rpref)
        self.shift = x86_shift(asmwrap=self.asmwrap, rpref=self.rpref)
        self.set = avx_opset(asmwrap=self.asmwrap, rpref=self.rpref, evex=True, simd_bytes=self.simd_size,
                              index_table=self.index_table)
        # the operations exist: the features that are off hide their signatures
        self._gate_features()

    def get_req_flags(self) -> list[str]:
        return ['avx512f']
    def max_fregs(self):
        return 32
    def max_vregs(self):
        return 32
    def mreg(self, reg_idx : int):
        """
        An AVX-512 mask register, k0..k7
        """
        return avx512_mreg(reg_idx)

    def shared_registers(self):
        """
        Floating-point registers are vector registers; masks are k registers,
        their own
        """
        return [("freg", "vreg", None)]

    def reserved_registers(self) -> dict[str, dict[str, int]]:
        """
        AVX2's, and k0, which can't be a write mask
        """
        regs = super().reserved_registers()
        regs["mreg"] = {"k0": 0}
        return regs

    def max_mregs(self):
        return 8

    @property
    def simd_size(self):
        return 64



    def zero_vreg(self, *, vreg : vreg_base, dt : adt):
        return self.set(dreg=vreg, dt=dt, modifiers={_opset_mod.CONST}, constant="zero")

    def zero_freg(self, *, freg : freg_base, dt : adt) -> str:
        # EVEX: FP scalars live in the vector file, allocated from the top -
        # xmm16-31 VEX's vpxor can't reach (AVX512VL)
        return self.set(dreg=freg, dt=dt, modifiers={_opset_mod.CONST}, constant="zero")

    def vreg(self, reg_idx : int):
        return zmm_vreg(reg_idx)

    def _k2_vindex(self, action : str, *, areg : greg_base, offvreg : vreg_base,
                   vreg : vreg_base, dt : adt, it : asm_index_type) -> str:
        """
        The old gather and scatter: k2 set all-true - the instruction clears
        it - then the access masked by it, as the old methods did
        """
        # pylint: disable=import-outside-toplevel
        from ..op.opmem import opmem_modifier
        from ..op.opset import opset_modifier
        k2 = self.mreg(2)
        try:
            access = getattr(self, action)(dregs=[vreg], areg=areg, dt=dt, it=it, amreg=k2, vidxreg=offvreg,
                                           modifiers={opmem_modifier.MASK, opmem_modifier.VINDEX})
        except ValueError as err:
            raise NotImplementedError(f"avx512 has no such {action}: {err}") from err
        return self.set(dreg=k2, dt=dt, modifiers={opset_modifier.CONSTMASK}, mask="alltrue") + access

    def load_vector_gather(self, *, areg : greg_base, offvreg : vreg_base,
                           vreg : vreg_base, dt : adt, it : asm_index_type) -> str:
        return self._k2_vindex("load", areg=areg, offvreg=offvreg, vreg=vreg, dt=dt, it=it)

    def store_vector_scatter(self, *, areg : greg_base, offvreg : vreg_base,
                             vreg : vreg_base, dt : adt, it : asm_index_type) -> str:
        return self._k2_vindex("store", areg=areg, offvreg=offvreg, vreg=vreg, dt=dt, it=it)

