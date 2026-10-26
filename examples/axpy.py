# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Example generating an AXPY function with dynamic unrolling and arithmetic selection.
"""

import importlib
import argparse

from asmgen.registers import (
    adt_size,
    asm_data_type as adt,
    reg_tracker
)
from asmgen.asmblocks.noarch import asmgen
from asmgen.callconv.fngen import fngen
from asmgen.asmblocks.noarch import comparison

from asmgen.asmblocks.op import (
    opmem_modifier as mem_mod,
    operand_modifier as opd_mod
)
from asmgen.asmblocks.op.operand import register_type as op_rt
from asmgen.asmblocks.op.operand import operand_modifier
from asmgen.asmblocks.op.opd3 import opd3_modifier
from asmgen.asmblocks.op.opset import opset_modifier

ISA_MODULES = {
    "fma128": "avx",
    "fma256": "avx",
    "avx512": "avx",
    "neon":   "neon",
    "sve":    "sve",
    "sme":    "sme",
    "rvv":    "rvv"
}

# -----------------------------------------------------------------------------
# Signature Resolution Helpers
# -----------------------------------------------------------------------------

def get_complexity(signature):
    """
    Sorts signatures by fewest instruction mods, fewest operand mods,
    then fewest operands
    """
    opd_mod_count = sum(len(shape.modifiers)
                        for shape in signature.operands.values()
                        if shape.modifiers)
    return (len(signature.modifiers), opd_mod_count, len(signature.operands))

def resolve_ldst_sig(signatures, dt, req_rt):
    # Only allow signatures that use operands this specific microkernel can provide
    safelist = {'adreg', 'agreg', 'amreg', 'ioffset', 'voffset'}

    valid = [s for s in signatures
             if s.operands['adreg'].dt == dt and s.operands['adreg'].rtype == req_rt]
    valid = [s for s in valid
             if set(s.operands.keys()).issubset(safelist)]

    if not valid:
        raise RuntimeError(f"No valid load/store signature found for {dt}")

    valid = sorted(valid, key=get_complexity)

    for s in valid:
        if mem_mod.VOFFSET in s.modifiers: return s, "voffset"
    for s in valid:
        if mem_mod.IOFFSET in s.modifiers: return s, "ioffset"

    return valid[0], "none"

def resolve_arith_sig(signatures, dt, req_rt):
    # AXPY only provides standard A, B, C, and Masks.
    # This automatically filters out BLOCKLANE, PART, and other exotic variants!
    safelist = {'adreg', 'bdreg', 'cdreg', 'amreg', 'bmreg', 'cmreg'}

    valid = [s for s in signatures
             if s.operands['adreg'].dt == dt and s.operands['adreg'].rtype == req_rt]
    valid = [s for s in valid if set(s.operands.keys()).issubset(safelist)]

    if not valid:
        raise RuntimeError(f"No valid arithmetic signature found for {dt}")

    valid = sorted(valid, key=get_complexity)

    for s in valid:
        if 'bdreg' in s.operands and opd_mod.VF in s.operands['bdreg'].modifiers:
            return s, True

    return valid[0], False

def build_kwargs(signature, mreg=None):
    """
    Supplies kwargs strictly based on what the signature's physical shape demands.
    """
    kw = {}
    if mreg is not None:
        for name, shape in signature.operands.items():
            if shape.rtype == op_rt.MASK:
                kw[name] = mreg
    return kw

def emit_ptr_bumping(gen: asmgen, addr_x, addr_y, vlen_reg, unroll, vbytes, dt):
    block = ""
    if vlen_reg:
        for _ in range(unroll):
            block += gen.add(adreg=addr_x, bdreg=vlen_reg, cdreg=addr_x, a_dt=adt.UINT64, b_dt=adt.UINT64, c_dt=adt.UINT64)
            block += gen.add(adreg=addr_y, bdreg=vlen_reg, cdreg=addr_y, a_dt=adt.UINT64, b_dt=adt.UINT64, c_dt=adt.UINT64)
    elif gen.add.fits(adreg=addr_x, bimm=unroll, cdreg=addr_x, a_dt=adt.UINT64, c_dt=adt.UINT64,
                      modifiers={opd3_modifier.VOFFSET}):
        # unroll vector lengths in one add, where the ISA has one (SVE's addvl)
        block += gen.add(adreg=addr_x, bimm=unroll, cdreg=addr_x, a_dt=adt.UINT64, c_dt=adt.UINT64, modifiers={opd3_modifier.VOFFSET})
        block += gen.add(adreg=addr_y, bimm=unroll, cdreg=addr_y, a_dt=adt.UINT64, c_dt=adt.UINT64, modifiers={opd3_modifier.VOFFSET})
    else:
        block += gen.add(adreg=addr_x, bimm=unroll * vbytes, cdreg=addr_x, a_dt=adt.UINT64, c_dt=adt.UINT64)
        block += gen.add(adreg=addr_y, bimm=unroll * vbytes, cdreg=addr_y, a_dt=adt.UINT64, c_dt=adt.UINT64)
    return block

# -----------------------------------------------------------------------------
# Main Generator
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AXPY Microkernel Generator")
    parser.add_argument("--isa", type=str, default="rvv", choices=ISA_MODULES.keys())
    parser.add_argument("--unroll", type=int, default=1,
                        help="Number of vectors to unroll")
    parser.add_argument("--arith", type=str, choices=['fma', 'mul_add', 'fmul_fadd'], default='fma')
    args = parser.parse_args()
    if args.arith == 'fmul_fadd':
        import sys
        print("axpy: --arith fmul_fadd is deprecated, use mul_add", file=sys.stderr)
        args.arith = 'mul_add'

    dt = adt.FP64
    module_name = ISA_MODULES[args.isa]

    vbytes_map = {'avx': 32
                  if args.isa == 'fma256' else (16 if args.isa == 'fma128' else 64),
                  'neon': 16}
    vbytes = vbytes_map.get(module_name, 16)

    generator_class = getattr(
            importlib.import_module(f"asmgen.asmblocks.{module_name}"), args.isa)
    gen: asmgen = generator_class()
    gen.set_output_inline(False)

    rt = gen.get_reg_tracker()

    cc = gen.create_callconv()
    cc.add_param("greg", "x")
    cc.add_param("greg", "y")
    cc.add_param("freg", "alpha", adt.FP64)
    cc.add_param("greg", "n")

    func = fngen(gen=gen, rt=rt)
    func.init_cc(cc=cc)

    addr_x = gen.greg(rt.aliased_regs["greg"]["x"])
    addr_y = gen.greg(rt.aliased_regs["greg"]["y"])
    n = gen.greg(rt.aliased_regs["greg"]["n"])
    alpha = gen.freg(rt.aliased_regs["freg"]["alpha"], dt=dt)

    x_regs = [gen.vreg(rt.reserve_any_reg("vreg")) for _ in range(args.unroll)]
    y_regs = [gen.vreg(rt.reserve_any_reg("vreg")) for _ in range(args.unroll)]

    innerblock = gen.isaquirks(dt=dt, rt=rt)

    vlen_reg = None
    if "vlen" in rt.aliased_regs["greg"]:
        vlen_reg = gen.greg(rt.aliased_regs["greg"]["vlen"])
        shift_amt = adt_size(dt).bit_length() - 1
        if shift_amt > 0:
            innerblock += gen.shift(adreg=vlen_reg, bimm=shift_amt, cdreg=vlen_reg, a_dt=adt.UINT64, c_dt=adt.UINT64, modifiers={opd3_modifier.SHL})

    # 1. Resolve Signatures
    ld_sig, offset_type = resolve_ldst_sig(gen.load.get_signatures(), dt, op_rt.VEC)
    st_sig, _ = resolve_ldst_sig(gen.store.get_signatures(), dt, op_rt.VEC)

    if args.arith == 'fma':
        arith_sig, can_vf = resolve_arith_sig(gen.fma.get_signatures(), dt, op_rt.VEC)
        arith_sig2 = None
    else:
        arith_sig, can_vf = resolve_arith_sig(gen.mul.get_signatures(), dt, op_rt.VEC)
        arith_sig2, _ = resolve_arith_sig(gen.add.get_signatures(), dt, op_rt.VEC)

    # 2. Setup Masking
    needs_mask = any(
        shape.rtype == op_rt.MASK
        for s in (ld_sig, st_sig, arith_sig, arith_sig2) if s
        for shape in s.operands.values()
    )

    mreg = None
    if needs_mask:
        mreg = gen.mreg(rt.reserve_any_reg("mreg"))
        # every lane on, where the ISA sets masks so (SVE's ptrue)
        if gen.set.forms(modifiers={opset_modifier.CONSTMASK}):
            innerblock += gen.set(dreg=mreg, dt=dt, modifiers={opset_modifier.CONSTMASK},
                                  mask="alltrue")

    ld_kwargs = build_kwargs(ld_sig, mreg)
    st_kwargs = build_kwargs(st_sig, mreg)

    if not can_vf:
        alpha_vreg = gen.vreg(rt.reserve_any_reg("vreg"))
        innerblock += gen.move(dregs=[alpha, alpha_vreg], dts=[dt, dt], operand_modifiers={"bdreg": {operand_modifier.BCAST}})
        breg = alpha_vreg
        opd_modifiers = {}
    else:
        breg = alpha
        opd_modifiers = {'bdreg': {opd_mod.VF}}

    innerblock += gen.label(label="loop")

    # 3. Core Loop Gen
    if offset_type != "none":
        for i in range(args.unroll):
            kw = ld_kwargs.copy()
            if offset_type == "voffset": kw['voffset'] = i
            else:                        kw['ioffset'] = i * vbytes
            innerblock += gen.load(dregs=[x_regs[i]], areg=addr_x,
                                   dt=dt, modifiers=ld_sig.modifiers, **kw)
            innerblock += gen.load(dregs=[y_regs[i]], areg=addr_y,
                                   dt=dt, modifiers=ld_sig.modifiers, **kw)

        for i in range(args.unroll):
            if args.arith == 'fma':
                innerblock += gen.fma(adreg=x_regs[i], bdreg=breg, cdreg=y_regs[i],
                                      a_dt=dt, b_dt=dt, c_dt=dt,
                                      modifiers=arith_sig.modifiers,
                                      operand_modifiers=opd_modifiers,
                                      **build_kwargs(arith_sig, mreg))
            else:
                innerblock += gen.mul(adreg=x_regs[i], bdreg=breg, cdreg=x_regs[i],
                                       a_dt=dt, b_dt=dt, c_dt=dt,
                                       modifiers=arith_sig.modifiers,
                                       operand_modifiers=opd_modifiers,
                                       **build_kwargs(arith_sig, mreg))
                innerblock += gen.add(adreg=x_regs[i], bdreg=y_regs[i], cdreg=y_regs[i],
                                       a_dt=dt, b_dt=dt, c_dt=dt,
                                       modifiers=arith_sig2.modifiers,
                                       operand_modifiers={},
                                       **build_kwargs(arith_sig2, mreg))

        for i in range(args.unroll):
            kw = st_kwargs.copy()
            if offset_type == "voffset": kw['voffset'] = i
            else:                        kw['ioffset'] = i * vbytes
            innerblock += gen.store(dregs=[y_regs[i]], areg=addr_y,
                                    dt=dt, modifiers=st_sig.modifiers, **kw)

        innerblock += emit_ptr_bumping(gen, addr_x, addr_y,
                                       vlen_reg, args.unroll, vbytes, dt)

    else:
        for i in range(args.unroll):
            innerblock += gen.load(dregs=[x_regs[i]], areg=addr_x,
                                   dt=dt, modifiers=ld_sig.modifiers, **ld_kwargs)
            innerblock += gen.load(dregs=[y_regs[i]], areg=addr_y,
                                   dt=dt, modifiers=ld_sig.modifiers, **ld_kwargs)

            if args.arith == 'fma':
                innerblock += gen.fma(adreg=x_regs[i], bdreg=breg, cdreg=y_regs[i],
                                      a_dt=dt, b_dt=dt, c_dt=dt,
                                      modifiers=arith_sig.modifiers,
                                      operand_modifiers=opd_modifiers,
                                      **build_kwargs(arith_sig, mreg))
            else:
                innerblock += gen.mul(adreg=x_regs[i], bdreg=breg, cdreg=x_regs[i],
                                       a_dt=dt, b_dt=dt, c_dt=dt,
                                       modifiers=arith_sig.modifiers,
                                       operand_modifiers=opd_modifiers,
                                       **build_kwargs(arith_sig, mreg))
                innerblock += gen.add(adreg=x_regs[i], bdreg=y_regs[i], cdreg=y_regs[i],
                                       a_dt=dt, b_dt=dt, c_dt=dt,
                                       modifiers=arith_sig2.modifiers,
                                       operand_modifiers={},
                                       **build_kwargs(arith_sig2, mreg))

            innerblock += gen.store(dregs=[y_regs[i]], areg=addr_y,
                                    dt=dt, modifiers=st_sig.modifiers, **st_kwargs)
            innerblock += emit_ptr_bumping(gen, addr_x, addr_y,
                                           vlen_reg, 1, vbytes, dt)

    innerblock += gen.add(adreg=n, bimm=-1, cdreg=n, a_dt=adt.UINT64, c_dt=adt.UINT64)
    innerblock += gen.cb(reg1=n, reg2=None, cmp=comparison.NZ, label="loop")

    if hasattr(gen, "isaendquirks"):
        innerblock += gen.isaendquirks(dt=dt, rt=rt)

    fnsave, fnload, fnrestore = func.get_boilerplate(cc=cc)

    print(".section .text")
    print(".global myfunction")
    print("myfunction:")
    print(fnsave + fnload + innerblock + fnrestore)

if __name__ == "__main__":
    main()
