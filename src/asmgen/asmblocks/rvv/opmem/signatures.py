# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Valid signatures for RVV opmem operations
"""

from dataclasses import dataclass
from typing import Callable

from ...op import (
    operation_signature as sig,
    operand_shape as osh,
    operand_type as ot,
    operand_role as orl,
    register_type as rt,
    opmem_modifier as mod,
    operand_modifier as opd_mod
)

from ...op.constraint import otherplusn_constraint
from ...op.misc import make_ord_prefix as mop

from ..types import rvv_vreg

from ....registers import (
    asm_data_type as adt,
    asm_index_type as ait,
    adt_size
)

@dataclass(kw_only=True)
class struct_constraint(otherplusn_constraint):
    """
    Constraint requiring segmented ld/st to use consecutive registers
    """
    what : str = 'index'
    getint : Callable[[rvv_vreg],int] = lambda reg : reg.idx
    makeval : Callable[[int],rvv_vreg] = lambda idx : rvv_vreg(reg_idx=idx)


def make_rvv_opmem_signatures(get_lmul: Callable[[],int], has_bcast=False):
    """
    Generate signatures for RVV opmem operations
    """

    sigs = []
    floats = [adt.FP64, adt.FP32, adt.FP16, adt.FP8E4M3, adt.FP8E5M2]
    ints = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8,
            adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]

    # VINDEX -> vidxreg, it
    # STRUCT -> nstructs
    # GSTRIDE -> streg


    index_type_map = {
        8 : ait.INT64,
        4 : ait.INT32,
        2 : ait.INT16,
        1 : ait.INT8
    }
    vidx_type_map = {
        8 : adt.SINT64,
        4 : adt.SINT32,
        2 : adt.SINT16,
        1 : adt.SINT8
    }

    for dt in floats+ints:
        # normal load/store
        sigs.append(sig(
            modifiers=set(),
            structural_params={},
            operands={
                'adreg': osh(ot.REGISTER, orl.DATA, rt.VEC, dt),
                'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64),
                }
        ))

        # STRUCT

        sc=struct_constraint
        # vlseg1e32 would be equivalent to vle32, but maybe some
        # selection logic could be simplified if we support it
        # explicitly
        for nstructs in range(1,9):
            if nstructs * get_lmul() > 8:
                # EMUL * NF <= 8: at m2 four fields, at m4 two
                continue
            operands = {
                'agreg' : osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64),
                'adreg' : osh(ot.REGISTER, orl.DATA, rt.VEC, dt)}
            operands.update({
                mop(i)+'dreg': osh(ot.REGISTER, orl.DATA, rt.VEC, dt,
                                   value_constraints=[
                                       sc(other=mop(i-1)+'dreg',
                                          offset=get_lmul())
                                       ])
                for i in range(1,nstructs)
                })
            sigs.append(sig(
                modifiers={mod.STRUCT},
                structural_params={
                    'nstructs': nstructs
                    },
                operands=operands
                ))


            if has_bcast:
                # BCAST + STRUCT
                bcast_operands = {
                    'agreg' : osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64),
                    'adreg' : osh(ot.REGISTER, orl.DATA, rt.VEC, dt, modifiers={opd_mod.BCAST})
                }
                bcast_operands.update({
                    mop(i)+'dreg': osh(ot.REGISTER, orl.DATA, rt.VEC, dt,
                                       modifiers={opd_mod.BCAST},
                                       value_constraints=[
                                           sc(other=mop(i-1)+'dreg', offset=get_lmul())
                                       ])
                    for i in range(1,nstructs)
                })
                sigs.append(sig(
                    modifiers={mod.STRUCT},
                    structural_params={'nstructs': nstructs},
                    operands=bcast_operands
                ))

        # VINDEX: byte offsets, of any width up to the data's (EMUL <= LMUL;
        # wider ones would make the index a register group)
        for isize, it in index_type_map.items():
            if isize > adt_size(dt):
                continue
            sigs.append(sig(
                modifiers={mod.VINDEX},
                structural_params={
                    'it': it
                    },
                operands={
                    'adreg': osh(ot.REGISTER, orl.DATA, rt.VEC, dt),
                    'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64),
                    'vidxreg': osh(ot.REGISTER, orl.ADDRESS, rt.VEC,
                                   vidx_type_map[isize])
                    }
                ))
        # GSTRIDE
        sigs.append(sig(
            modifiers={mod.GSTRIDE},
            structural_params={
                },
            operands={
                'adreg': osh(ot.REGISTER, orl.DATA, rt.VEC, dt),
                'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64),
                'streg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64)
                }
            ))

        if has_bcast:
            # BCAST
            sigs.append(sig(
                modifiers=set(),
                structural_params={
                    },
                operands={
                    'adreg': osh(ot.REGISTER, orl.DATA, rt.VEC, dt,
                                 modifiers={opd_mod.BCAST}),
                    'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64)
                    }
                ))

        # STRUCT + GSTRIDE: segments streg bytes apart (vlsseg<nf>, vssseg<nf>) -
        # a panel's rows packed in parts, each part's rows interleaved into
        # the panel's columns cdim apart
        for nstructs in range(2, 9):
            if nstructs * get_lmul() > 8:
                continue
            operands = {
                'agreg' : osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64),
                'streg' : osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64),
                'adreg' : osh(ot.REGISTER, orl.DATA, rt.VEC, dt)}
            operands.update({
                mop(i)+'dreg': osh(ot.REGISTER, orl.DATA, rt.VEC, dt,
                                   value_constraints=[sc(other=mop(i-1)+'dreg', offset=get_lmul())])
                for i in range(1, nstructs)})
            sigs.append(sig(
                modifiers={mod.STRUCT, mod.GSTRIDE},
                structural_params={'nstructs': nstructs},
                operands=operands))

        # MASK: the elements v0 holds (vle64.v vd, (rs1), v0.t) - a last
        # vector's first elements; unit-stride and strided
        for mods, extra in (({mod.MASK}, {}),
                            ({mod.MASK, mod.GSTRIDE},
                             {'streg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64)})):
            sigs.append(sig(
                modifiers=mods,
                structural_params={},
                operands={'adreg': osh(ot.REGISTER, orl.DATA, rt.VEC, dt),
                          'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64),
                          'amreg': osh(ot.REGISTER, orl.MASK, rt.VEC, dt),
                          **extra}))

        # TODO:
        # - STRUCT+VINDEX

        # TODO (also in the opmem implementation)
        # - everything + MASK


    return sigs
