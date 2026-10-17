# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Signatures for AVX move instructions
"""

from ...op import (
    operation_signature as opsig,
    operand_modifier as omod,
    operand_shape as osh,
    operand_type as ot,
    operand_role as orl,
    register_type as rgt,
)
from ...op.constraint import minmax_constraint
from ....registers import asm_data_type as adt, adt_size


def make_avx_move_signatures(simd_bytes : int) -> list[opsig]:
    """
    adreg is the source, bdreg the destination; blocks are 128 bits:

        copy             vmovaps
        scalar to all    vbroadcastsd/ss (vmovddup for doubles in xmm)
        block to all     vshuff32x4 (zmm), vperm2f128 (ymm)   adreg IBLOCK, bdreg BLOCK+BCAST
        block out        vextractf32x4, vextractf128           adreg IBLOCK
        block in         vinsertf32x4, vinsertf128             bdreg IBLOCK
    """
    sigs = []
    blocks = simd_bytes // 16

    def add_sig(dt, amods, bmods):
        ops = {
            'adreg': osh(otype=ot.REGISTER, orole=orl.DATA,
                         rtype=rgt.FP if (omod.BCAST in bmods and not amods) else rgt.VEC, dt=dt),
            'bdreg': osh(otype=ot.REGISTER, orole=orl.DATA, rtype=rgt.VEC, dt=dt),
        }
        params = {}
        for name, mods in (('adreg', amods), ('bdreg', bmods)):
            if omod.IBLOCK in mods:
                ops[f'{name}_block'] = osh(otype=ot.IMMEDIATE, orole=orl.PARAM, value_constraints=[
                    minmax_constraint(minval=0, maxval=blocks - 1)])
            if {omod.IBLOCK, omod.BLOCK} & mods:
                params[f'{name}_blocksize'] = 16 // adt_size(dt)
        ops['adreg'].modifiers = amods
        ops['bdreg'].modifiers = bmods
        sigs.append(opsig(modifiers=set(), structural_params=params, operands=ops))

    for dt in (adt.FP64, adt.FP32):
        add_sig(dt, set(), set())
        add_sig(dt, set(), {omod.BCAST})
        if blocks > 1:
            add_sig(dt, {omod.IBLOCK}, {omod.BLOCK, omod.BCAST})
            add_sig(dt, {omod.IBLOCK}, set())
            add_sig(dt, set(), {omod.IBLOCK})
    return sigs
