# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Signatures for SME move instructions
"""

from ...op.constraint import minmax_constraint
from typing import Type

from dataclasses import dataclass

from ...op import (
    operation_signature as opsig,
    operand_modifier as omod,
    operand_shape as osh,
    operand_type as ot,
    operand_role as orl,
    register_type as rgt,
    opmove_modifier as mod,
    make_ord_prefix as mop
)

from ...op.constraint import (
    otherplusn_constraint,
    otherplusnmod_constraint,
    oneof_constraint,
    regidx_constraint,
    sameval_constraint
)

from ....registers import asm_data_type as adt,adt_size

from ...aarch64.types import aarch64_greg
from ...sve.types import sve_vreg

from ...sme.opmem.signatures import sme_rowcolreg_constraint

_FLOATS = [adt.FP64, adt.FP32, adt.FP16, adt.BF16, adt.FP8E4M3, adt.FP8E5M2]
_INTS = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8,
         adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]


@dataclass(kw_only=True)
class samegregidx_constraint(regidx_constraint,
                             sameval_constraint):
    """
    Ensure the same register is used for all rows/cols
    """
    reg_class : Type[aarch64_greg] = aarch64_greg

@dataclass(kw_only=True)
class consecutive_vregs_constraint(regidx_constraint,
                                   otherplusnmod_constraint):
    """
    Ensure vregs are consecutive when inserting/extracting multiple
    rows/columns
    """
    offset : int = 1
    modval : int = 32
    reg_class : Type[sve_vreg] = sve_vreg


#TODO: modularize/break up/generalize/deduplicate
def make_sme_move_signatures() -> list[opsig]:
    """
    Generate signatures for SME data move instructions
    """

    sigs = []


    def add_sig(dt: adt, rcmod :omod, t_to_v : bool,
                inputs : int = 1,
                outputs : int = 1):
        rcstr = "col" if rcmod == omod.COL else "row"

        structural_params = {}

        operands = {
                'amreg' : osh(otype=ot.REGISTER, orole=orl.MASK,
                              rtype=rgt.MASK, dt=dt)
            }
        if t_to_v:
            # 4 slices exist for 64bit (SVL >256bit required)
            # 4 slices exist for 32bit
            # 8 slices exist for 16bit
            # 16 slices exist for 8 bit
            slices = max(4, 16//adt_size(dt))

            # #inputs = #outputs
            nvecs = inputs

            operands['adreg'] = osh(
                    otype=ot.REGISTER,
                    orole=orl.DATA,
                    rtype=rgt.TILE,
                    dt=dt,
                    modifiers={rcmod}
                    )
            operands[f'adreg_imm{rcstr}'] = osh(
                    otype=ot.IMMEDIATE,
                    orole=orl.PARAM,
                    value_constraints=[oneof_constraint(
                        valset={i*nvecs for i in range(slices//nvecs)})]
                    )
            operands[f'adreg_{rcstr}reg'] = osh(
                    otype=ot.REGISTER,
                    orole=orl.PARAM,
                    rtype=rgt.GP,
                    value_constraints=[sme_rowcolreg_constraint()]
                    )
            for i in range(1,inputs):
                rc_treg = f"{mop(i)}dreg"
                operands[rc_treg] = osh(
                        otype=ot.REGISTER,
                        orole=orl.DATA,
                        rtype=rgt.TILE,
                        dt=dt,
                        modifiers={rcmod}
                        )
                operands[f'{rc_treg}_imm{rcstr}'] = osh(
                        otype=ot.IMMEDIATE,
                        orole=orl.PARAM,
                        value_constraints=[otherplusn_constraint(
                            other=f"{mop(i-1)}dreg_imm{rcstr}",
                            offset=1
                            )]
                        )
                operands[f'{rc_treg}_{rcstr}reg'] = osh(
                        otype=ot.REGISTER,
                        orole=orl.PARAM,
                        rtype=rgt.GP,
                        value_constraints=[
                            samegregidx_constraint(
                                other=f"{mop(inputs)}dreg_{rcstr}reg")
                            ])

            for i in range(outputs):
                vreg = f"{mop(i+inputs)}dreg"
                operands[vreg] = osh(
                        otype=ot.REGISTER,
                        orole=orl.DATA,
                        rtype=rgt.VEC,
                        dt=dt
                        )
            for i in range(1,outputs):
                vreg = f"{mop(i+inputs)}dreg"
                pre_vreg = f"{mop(i+inputs-1)}dreg"
                operands[vreg].value_constraints = [
                    consecutive_vregs_constraint(other=pre_vreg)
                ]
        else:
            # Vectors are inputs (adreg, bdreg, ...)
            for i in range(inputs):
                vreg = f"{mop(i)}dreg"
                operands[vreg] = osh(
                        otype=ot.REGISTER,
                        orole=orl.DATA,
                        rtype=rgt.VEC,
                        dt=dt
                        )

            # Constraint: sequential vector registers
            for i in range(1, inputs):
                vreg = f"{mop(i)}dreg"
                pre_vreg = f"{mop(i-1)}dreg"
                operands[vreg].value_constraints = [
                    consecutive_vregs_constraint(other=pre_vreg)
                ]

            # Tiles are outputs (e.g., edreg, fdreg, gdreg, hdreg for inputs=4)
            for i in range(outputs):
                treg = f"{mop(i+inputs)}dreg"
                operands[treg] = osh(
                        otype=ot.REGISTER,
                        orole=orl.DATA,
                        rtype=rgt.TILE,
                        dt=dt,
                        modifiers={rcmod}
                        )

                if i == 0:
                    # the first slice: the tile has 16 bytes' worth of slices
                    # per element size, the following ones after it
                    operands[f'{treg}_imm{rcstr}'] = osh(
                            otype=ot.IMMEDIATE,
                            orole=orl.PARAM,
                            value_constraints=[minmax_constraint(
                                minval=0, maxval=max(0, 16 // adt_size(dt) - outputs))])
                    operands[f'{treg}_{rcstr}reg'] = osh(
                            otype=ot.REGISTER,
                            orole=orl.PARAM,
                            rtype=rgt.GP,
                            value_constraints=[sme_rowcolreg_constraint()]
                            )
                else:
                    operands[f'{treg}_imm{rcstr}'] = osh(
                            otype=ot.IMMEDIATE,
                            orole=orl.PARAM,
                            value_constraints=[otherplusn_constraint(
                                other=f"{mop(i+inputs-1)}dreg_imm{rcstr}",
                                offset=1
                                )]
                            )
                    operands[f'{treg}_{rcstr}reg'] = osh(
                            otype=ot.REGISTER,
                            orole=orl.PARAM,
                            rtype=rgt.GP,
                            value_constraints=[
                                samegregidx_constraint(
                                    other=f"{mop(inputs)}dreg_{rcstr}reg")
                                ])



        modifiers={mod.MASK}

        if inputs > 1:
            structural_params['nin'] = inputs
            modifiers.add(mod.MULTIPLE_IN)
        if outputs > 1:
            structural_params['nout'] = outputs
            modifiers.add(mod.MULTIPLE_OUT)

        sigs.append(
            opsig(modifiers=modifiers,
                  structural_params=structural_params,
                  operands=operands)
                )

    for dt in _FLOATS+_INTS:
        add_sig(dt=dt, rcmod=omod.COL, t_to_v=False)
        add_sig(dt=dt, rcmod=omod.ROW, t_to_v=False)
        add_sig(dt=dt, rcmod=omod.COL, t_to_v=True)
        add_sig(dt=dt, rcmod=omod.ROW, t_to_v=True)
        add_sig(dt=dt, rcmod=omod.COL, t_to_v=False, inputs=2, outputs=2)
        add_sig(dt=dt, rcmod=omod.ROW, t_to_v=False, inputs=2, outputs=2)
        add_sig(dt=dt, rcmod=omod.COL, t_to_v=False, inputs=4, outputs=4)
        add_sig(dt=dt, rcmod=omod.ROW, t_to_v=False, inputs=4, outputs=4)
        add_sig(dt=dt, rcmod=omod.COL, t_to_v=True, inputs=2, outputs=2)
        add_sig(dt=dt, rcmod=omod.ROW, t_to_v=True, inputs=2, outputs=2)
        add_sig(dt=dt, rcmod=omod.COL, t_to_v=True, inputs=4, outputs=4)
        add_sig(dt=dt, rcmod=omod.ROW, t_to_v=True, inputs=4, outputs=4)

    return sigs
