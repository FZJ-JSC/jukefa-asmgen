# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
data move operation for AVX
"""

from typing import Callable

from ....registers import asm_data_type as adt, data_reg
from ...op.signature import operation_signature
from ...op.operand import operand_modifier as omod
from ...op.move import move, move_modifier as mod
from ..types import xmm_vreg, ymm_vreg
from .signatures import make_avx_move_signatures


class avx_move(move):
    """
    AVX data move instructions on 128-bit blocks, in AT&T operand order. The
    block forms move bits, so their 32x4 variants serve every element type.
    """

    def __init__(self, asmwrap : Callable[[str], str], rpref, simd_bytes : int):
        self.asmwrap = asmwrap
        self.rpref = rpref
        self.simd_bytes = simd_bytes
        self.signatures = make_avx_move_signatures(simd_bytes)

    def get_signatures(self) -> list[operation_signature]:
        return self.signatures

    def implementation(self, *, dregs : list[data_reg],
                       dts : dict[str, adt],
                       modifiers : set[mod],
                       operand_modifiers : dict[str, set[omod]],
                       **kwargs) -> str:
        src, dst = dregs[0], dregs[1]
        dt = dts['bdreg']
        amods = operand_modifiers.get('adreg', set())
        bmods = operand_modifiers.get('bdreg', set())
        p, wide = self.rpref, self.simd_bytes == 64
        x = lambda r: p(xmm_vreg(r.idx))

        if omod.IBLOCK in amods and omod.BCAST in bmods:
            i = kwargs['adreg_block']
            if wide:
                imm = i * 0b01010101
                return self.asmwrap(f"vshuff32x4 ${imm:#x},{p(src)},{p(src)},{p(dst)}")
            return self.asmwrap(f"vperm2f128 ${i | (i << 4):#x},{p(src)},{p(src)},{p(dst)}")
        if omod.IBLOCK in amods:
            inst = "vextractf32x4" if wide else "vextractf128"
            return self.asmwrap(f"{inst} ${kwargs['adreg_block']},{p(src)},{x(dst)}")
        if omod.IBLOCK in bmods:
            inst = "vinsertf32x4" if wide else "vinsertf128"
            return self.asmwrap(f"{inst} ${kwargs['bdreg_block']},{x(src)},{p(dst)},{p(dst)}")
        if omod.BCAST in bmods:
            if adt.FP64 == dt and 16 == self.simd_bytes:
                return self.asmwrap(f"vmovddup {x(src)},{p(dst)}")
            suf = "sd" if adt.FP64 == dt else "ss"
            return self.asmwrap(f"vbroadcast{suf} {x(src)},{p(dst)}")
        return self.asmwrap(f"vmovaps {p(src)},{p(dst)}")
