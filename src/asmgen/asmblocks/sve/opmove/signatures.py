
# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Signatures for SVE move instructions
"""

from ...op import (
    operation_signature as opsig,
    operand_modifier as omod,
    operand_shape as osh,
    operand_type as ot,
    operand_role as orl,
    register_type as rgt,
    opmove_modifier as mod,
)

from ...op.constraint import (
    minmax_constraint
)

from ....registers import asm_data_type as adt,adt_size
from ...op.shift import shift_signatures, fill_signatures, extract_signatures

_FLOATS = [adt.FP64, adt.FP32, adt.FP16, adt.BF16, adt.FP8E4M3, adt.FP8E5M2]
_INTS = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8,
         adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]


def make_sve_move_signatures() -> list[opsig]:
    """
    Generate signatures for SVE data move instructions
    """

    sigs = []


    def add_sig(dt: adt,
                mods : mod,
                amods : set[omod],
                bmods : set[omod]):

        structural_params = {}
        operands = {}

        if mod.MASK in mods:
            operands['amreg'] =  osh(otype=ot.REGISTER,
                                     orole=orl.MASK,
                                     rtype=rgt.MASK, dt=dt)


        operands['bdreg'] = osh(otype=ot.REGISTER,
                                orole=orl.DATA,
                                rtype=rgt.VEC, dt=dt)

        if not bmods and not amods:
            operands['adreg'] = osh(otype=ot.REGISTER,
                                    orole=orl.DATA,
                                    rtype=rgt.VEC, dt=dt)

        if (omod.BCAST in bmods) and not amods:
            if dt in _FLOATS:
                operands['adreg'] = osh(otype=ot.REGISTER,
                                        orole=orl.DATA,
                                        rtype=rgt.FP, dt=dt)
            else:
                operands['adreg'] = osh(otype=ot.REGISTER,
                                        orole=orl.DATA,
                                        rtype=rgt.GP, dt=dt)
        if omod.IBLOCK in amods:
            # dup z.q, z.q[i]: 128-bit blocks, at most four in 512 bits
            operands['adreg'] = osh(otype=ot.REGISTER, orole=orl.DATA, rtype=rgt.VEC, dt=dt)
            operands['adreg_block'] = osh(otype=ot.IMMEDIATE, orole=orl.PARAM,
                                          value_constraints=[minmax_constraint(minval=0, maxval=3)])
            structural_params['adreg_blocksize'] = 16 // adt_size(dt)
            structural_params['bdreg_blocksize'] = 16 // adt_size(dt)
        if omod.ILANE in amods:
            operands['adreg'] = osh(otype=ot.REGISTER,
                                    orole=orl.DATA,
                                    rtype=rgt.VEC, dt=dt)

            # 512 bits are directly addressable
            max_lane = 512 // 8 // adt_size(dt) - 1
            operands['adreg_lane'] = osh(otype=ot.IMMEDIATE,
                                         orole=orl.PARAM,
                                         value_constraints=[
                                             minmax_constraint(minval=0,maxval=max_lane)
                                             ])

        operands['adreg'].modifiers = amods
        operands['bdreg'].modifiers = bmods

        sigs.append(
            opsig(modifiers=mods,
                  structural_params=structural_params,
                  operands=operands)
                )

    for dt in _FLOATS+_INTS:
        add_sig(dt=dt, mods=set(), amods=set(), bmods=set())
        add_sig(dt=dt, mods=set(), amods=set(), bmods={omod.BCAST})
        add_sig(dt=dt, mods=set(), amods={omod.ILANE}, bmods={omod.BCAST})
        add_sig(dt=dt, mods=set(), amods={omod.IBLOCK}, bmods={omod.BLOCK, omod.BCAST})
        add_sig(dt=dt, mods={mod.MASK}, amods=set(), bmods=set())
        add_sig(dt=dt, mods={mod.MASK}, amods=set(), bmods={omod.BCAST})
        add_sig(dt=dt, mods={mod.MASK}, amods={omod.ILANE}, bmods={omod.BCAST})

    # destructive ext with the register twice: a rotation towards lane 0 by a
    # number of bytes known now - lanes or 16-byte blocks; insr shifts away
    # from lane 0 by one, a scalar into lane 0
    sigs += shift_signatures(
        _FLOATS + _INTS, [{omod.ILANEROT}, {omod.ILANESHIFT, omod.INVALIDOTHERS}],
        irot=lambda dt, m: (-(255 // adt_size(dt)), -1), destructive=True)
    sigs += shift_signatures(
        _FLOATS + _INTS, [{omod.IBLOCKROT}], irot=lambda dt, m: (-15, -1),
        destructive=True, blocksize=lambda dt: 16 // adt_size(dt))
    sigs += shift_signatures(
        [dt for dt in _FLOATS + _INTS if dt not in (adt.BF16, adt.FP8E4M3, adt.FP8E5M2)],
        [{omod.ILANESHIFT, omod.FILLOTHERS}], irot=lambda dt, m: (1, 1), destructive=True,
        fill=lambda dt: rgt.FP if dt in _FLOATS else rgt.GP)
    # a fill loop's step: the scalar into lane 0, the vector shifted up in
    # place (insr); a drain loop's: lane 0, the register's low part, out
    scalar = lambda dt: rgt.FP if dt in _FLOATS else rgt.GP
    plain = [dt for dt in _FLOATS + _INTS if dt not in (adt.BF16, adt.FP8E4M3, adt.FP8E5M2)]
    sigs += fill_signatures(plain, irot=lambda dt: 1, scalar=scalar)
    sigs += extract_signatures([dt for dt in plain if dt in _FLOATS], scalar=scalar)
    return sigs
