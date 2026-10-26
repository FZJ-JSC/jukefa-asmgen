# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Example showing how the LMUL parameter of the rvv generator works
"""
# Output should be:
#
# vsetvli t4, zero, e64, m1, ta, ma
# slli t4,t4,3
# vle64.v v0, (t0)
# vle64.v v1, (t1)
# vle64.v v2, (t2)
# fld f0, 0(t3)
# vfmacc.vv v2,v1,v0
# vfmacc.vf v2,f0,v0
# add t0,t0,t4
# add t1,t1,t4
# add t2,t2,t4
# add t3,t3,8
#
# vsetvli t4, zero, e64, m2, ta, ma
# slli t4,t4,3
# vle64.v v0, (t0)
# vle64.v v2, (t1)
# vle64.v v4, (t2)
# fld f0, 0(t3)
# vfmacc.vv v4,v2,v0
# vfmacc.vf v4,f0,v0
# add t0,t0,t4
# add t1,t1,t4
# add t2,t2,t4
# add t3,t3,8
#
# vsetvli t4, zero, e64, m4, ta, ma
# slli t4,t4,3
# vle64.v v0, (t0)
# vle64.v v4, (t1)
# vle64.v v8, (t2)
# fld f0, 0(t3)
# vfmacc.vv v8,v4,v0
# vfmacc.vf v8,f0,v0
# add t0,t0,t4
# add t1,t1,t4
# add t2,t2,t4
# add t3,t3,8
#
# vsetvli t4, zero, e64, m8, ta, ma
# slli t4,t4,3
# vle64.v v0, (t0)
# vle64.v v8, (t1)
# vle64.v v16, (t2)
# fld f0, 0(t3)
# vfmacc.vv v16,v8,v0
# vfmacc.vf v16,f0,v0
# add t0,t0,t4
# add t1,t1,t4
# add t2,t2,t4
# add t3,t3,8

from asmgen.asmblocks.rvv import rvv
from asmgen.registers import (
    adt_size,
    asm_data_type as adt
)
from asmgen.asmblocks.op.optransform import optransform_modifier
from asmgen.asmblocks.op.operand import operand_modifier


def main():
    gen = rvv()


    gen.set_output_inline(yesno=False)

    dt = adt.FP64

    for lmul in [1,2,4,8]:
        gen.set_parameter("LMUL", lmul)
        rt = gen.get_reg_tracker()

        aidx = rt.reserve_any_reg("vreg")
        a = gen.vreg(aidx)
        bidx = rt.reserve_any_reg("vreg")
        b = gen.vreg(bidx)
        cidx = rt.reserve_any_reg("vreg")
        c = gen.vreg(cidx)

        aaddr_idx = rt.reserve_any_reg("greg")
        aaddr = gen.greg(aaddr_idx)
        baddr_idx = rt.reserve_any_reg("greg")
        baddr = gen.greg(baddr_idx)
        caddr_idx = rt.reserve_any_reg("greg")
        caddr = gen.greg(caddr_idx)

        bfidx = rt.reserve_any_reg("freg")
        bf = gen.freg(bfidx,dt=dt)

        bfaddr_idx = rt.reserve_any_reg("greg")
        bfaddr = gen.greg(bfaddr_idx)


        asmblock = gen.isaquirks(dt=dt,rt=rt)

        vlenidx = rt.aliased_regs["greg"]["vlen"]
        vlen = gen.greg(vlenidx)

        asmblock += gen.shift(adreg=vlen, bimm=adt_size(dt).bit_length()-1, cdreg=vlen, a_dt=adt.UINT64, c_dt=adt.UINT64, modifiers={optransform_modifier.SHL})


        asmblock += gen.load(dregs=[a], areg=aaddr, dt=dt)
        asmblock += gen.load(dregs=[b], areg=baddr, dt=dt)
        asmblock += gen.load(dregs=[c], areg=caddr, dt=dt)

        asmblock += gen.load(dregs=[bf], areg=bfaddr, dt=dt)

        asmblock += gen.fma(
                adreg=a,
                bdreg=b,
                cdreg=c,
                a_dt=dt,
                b_dt=dt,
                c_dt=dt)
        asmblock += gen.fma(
                adreg=a,
                bdreg=bf,
                cdreg=c,
                a_dt=dt,
                b_dt=dt,
                c_dt=dt,
                operand_modifiers={"bdreg": {operand_modifier.VF}})   # bf a scalar: vfmacc.vf


        asmblock += gen.add(adreg=aaddr, bdreg=vlen, cdreg=aaddr, a_dt=adt.UINT64, b_dt=adt.UINT64, c_dt=adt.UINT64)
        asmblock += gen.add(adreg=baddr, bdreg=vlen, cdreg=baddr, a_dt=adt.UINT64, b_dt=adt.UINT64, c_dt=adt.UINT64)
        asmblock += gen.add(adreg=caddr, bdreg=vlen, cdreg=caddr, a_dt=adt.UINT64, b_dt=adt.UINT64, c_dt=adt.UINT64)

        asmblock += gen.add(adreg=bfaddr, bimm=adt_size(dt), cdreg=bfaddr, a_dt=adt.UINT64, c_dt=adt.UINT64)

        asmblock += gen.isaendquirks(dt=dt, rt=rt)

        rt.unuse_reg("vreg", aidx)
        rt.unuse_reg("vreg", bidx)
        rt.unuse_reg("vreg", cidx)
        rt.unuse_reg("freg", bfidx)

        print(asmblock)


if __name__ == "__main__":
    main()
