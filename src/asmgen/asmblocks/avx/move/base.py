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
from ....registers import adt_size
from .signatures import make_avx_move_signatures
from ...x86.move import x86_move
from ...op.scalar import scalar_dregs_only, scalar_operands


class avx_move(move):
    """
    AVX data move instructions on 128-bit blocks, in AT&T operand order. The
    block forms move bits, so their 32x4 variants serve every element type.
    """
    # the base ISA's moves between scalar registers: a member whose
    # signatures are offered too and to which calls with them go
    scalar_class = x86_move

    def __init__(self, asmwrap : Callable[[str], str], rpref, simd_bytes : int):
        self.asmwrap = asmwrap
        self.rpref = rpref
        self.simd_bytes = simd_bytes
        self.signatures = make_avx_move_signatures(simd_bytes)
        self.scalar_op = self.scalar_class(asmwrap=asmwrap, rpref=rpref) if self.scalar_class else None

    def get_signatures(self) -> list[operation_signature]:
        return [*self.signatures, *(self.scalar_op.get_signatures() if self.scalar_op else [])]

    def implementation(self, *, dregs : list[data_reg],
                       dts : dict[str, adt],
                       modifiers : set[mod],
                       operand_modifiers : dict[str, set[omod]],
                       **kwargs) -> str:
        # If scalar registers only are passed, forward to the base ISA's move
        if self.scalar_op is not None and scalar_dregs_only(dregs):
            return self.scalar_op(dregs=dregs, dts=[dts[k] for k in sorted(dts)],
                                  modifiers=modifiers, operand_modifiers=operand_modifiers,
                                  **scalar_operands(kwargs))

        src, dst = dregs[0], dregs[1]
        dt = dts['bdreg']
        amods = operand_modifiers.get('adreg', set())
        bmods = operand_modifiers.get('bdreg', set())
        p, wide = self.rpref, self.simd_bytes == 64
        block = lambda name: kwargs.get(f"{name}_blocksize", 0) * adt_size(dt)
        # the register holding one block: xmm for 128 bits, ymm for 256
        part = lambda r, size: p(ymm_vreg(r.idx) if 32 == size else xmm_vreg(r.idx))
        x = lambda r: p(xmm_vreg(r.idx))

        if {omod.ILANEROT, omod.ILANESHIFT} & amods:
            size = adt_size(dt)
            imm = (-kwargs['adreg_irot']) % (self.simd_bytes // size)
            inst = "valignq" if 8 == size else "valignd"
            return self.asmwrap(f"{inst} ${imm},{p(src)},{p(src)},{p(dst)}")
        if omod.IBLOCKROT in amods:
            r = (-kwargs['adreg_irot']) % 4
            imm = sum(((j + r) % 4) << (2 * j) for j in range(4))
            return self.asmwrap(f"vshuff64x2 ${imm},{p(src)},{p(src)},{p(dst)}")

        if omod.IBLOCK in amods and omod.BCAST in bmods:
            i = kwargs['adreg_block']
            if wide and 32 == block('adreg'):
                # both 128-bit halves of 256-bit block i, twice
                imm = (2 * i) | (2 * i + 1) << 2 | (2 * i) << 4 | (2 * i + 1) << 6
                return self.asmwrap(f"vshuff64x2 ${imm:#x},{p(src)},{p(src)},{p(dst)}")
            if wide:
                imm = i * 0b01010101
                return self.asmwrap(f"vshuff32x4 ${imm:#x},{p(src)},{p(src)},{p(dst)}")
            return self.asmwrap(f"vperm2f128 ${i | (i << 4):#x},{p(src)},{p(src)},{p(dst)}")
        if omod.IBLOCK in amods:
            size = block('adreg')
            inst = "vextractf64x4" if 32 == size else ("vextractf32x4" if wide else "vextractf128")
            return self.asmwrap(f"{inst} ${kwargs['adreg_block']},{p(src)},{part(dst, size)}")
        if omod.IBLOCK in bmods:
            size = block('bdreg')
            inst = "vinsertf64x4" if 32 == size else ("vinsertf32x4" if wide else "vinsertf128")
            return self.asmwrap(f"{inst} ${kwargs['bdreg_block']},{part(src, size)},{p(dst)},{p(dst)}")
        if omod.BCAST in bmods:
            if adt.FP64 == dt and 16 == self.simd_bytes:
                return self.asmwrap(f"vmovddup {x(src)},{p(dst)}")
            if adt.FP16 == dt:
                return self.asmwrap(f"vpbroadcastw {x(src)},{p(dst)}")
            suf = "sd" if adt.FP64 == dt else "ss"
            return self.asmwrap(f"vbroadcast{suf} {x(src)},{p(dst)}")
        return self.asmwrap(f"vmovaps {p(src)},{p(dst)}")
