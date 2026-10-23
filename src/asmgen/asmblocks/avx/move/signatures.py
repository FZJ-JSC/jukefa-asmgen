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
from ...op.shift import shift_signatures
from ....registers import asm_data_type as adt, adt_size


def make_avx_move_signatures(simd_bytes : int) -> list[opsig]:
    """
    adreg is the source, bdreg the destination; blocks are 128 bits, or on zmm
    also 256 bits (vshuff64x2, vextractf64x4, vinsertf64x4):

        copy             vmovaps
        scalar to all    vbroadcastsd/ss (vmovddup for doubles in xmm)
        block to all     vshuff32x4 (zmm), vperm2f128 (ymm)   adreg IBLOCK, bdreg BLOCK+BCAST
        block out        vextractf32x4, vextractf128           adreg IBLOCK
        block in         vinsertf32x4, vinsertf128             bdreg IBLOCK
    """
    sigs = []

    def add_sig(dt, amods, bmods, block_bytes=16, requires=frozenset()):
        blocks = simd_bytes // block_bytes
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
                params[f'{name}_blocksize'] = block_bytes // adt_size(dt)
        ops['adreg'].modifiers = amods
        ops['bdreg'].modifiers = bmods
        sigs.append(opsig(modifiers=set(), structural_params=params, operands=ops, requires=requires))

    for dt in (adt.FP64, adt.FP32):
        add_sig(dt, set(), set())
        add_sig(dt, set(), {omod.BCAST})
        # 128-bit blocks, and on zmm 256-bit ones
        for block_bytes in (16, 32):
            if simd_bytes // block_bytes < 2:
                continue
            add_sig(dt, {omod.IBLOCK}, {omod.BLOCK, omod.BCAST}, block_bytes)
            add_sig(dt, {omod.IBLOCK}, set(), block_bytes)
            add_sig(dt, set(), {omod.IBLOCK}, block_bytes)
    # FP16 from a register into every element: a 16-bit broadcast,
    # vpbroadcastw - AVX2's on xmm and ymm, AVX512BW's on zmm
    add_sig(adt.FP16, set(), {omod.BCAST}, requires=frozenset({'BW'}) if 64 == simd_bytes else frozenset())
    if 64 == simd_bytes:
        # valignq/valignd with the register twice: a rotation by lanes;
        # vshuff64x2 likewise by 128-bit blocks
        sigs += shift_signatures(
            [dt for dt in (adt.FP64, adt.FP32, adt.SINT64, adt.UINT64, adt.SINT32, adt.UINT32)],
            [{omod.ILANEROT}, {omod.ILANESHIFT, omod.INVALIDOTHERS}],
            irot=lambda dt, m: (1 - 64 // adt_size(dt), 64 // adt_size(dt) - 1))
        sigs += shift_signatures(
            [adt.FP64, adt.FP32], [{omod.IBLOCKROT}], irot=lambda dt, m: (-3, 3),
            blocksize=lambda dt: 16 // adt_size(dt))
    return sigs
