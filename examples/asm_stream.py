# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# Copyright (C) 2026 Daniel Seibel <d.seibel@fz-juelich.de>
# ------------------------------------------------------------------------------

import argparse
import sys
import importlib

from asmgen.registers import (
        adt_size,
        asm_data_type as adt,
        reg_tracker
)
from asmgen.callconv.fngen import fngen
from asmgen.asmblocks.noarch import comparison, asmgen
from asmgen.asmblocks.operations import modifier as mod

isa_modules = {
        "fma128" : "avx_fma",
        "fma256" : "avx_fma",
        "avx512" : "avx_fma",
}

def main():
    parser = argparse.ArgumentParser(description="Generate unrolled STREAM assembly across multiple ISAs.")
    parser.add_argument("--isa", type=str, default="rvv", help="Target ISA (e.g., rvv, fma256, avx512)")
    parser.add_argument("--op", choices=["copy", "scale", "add", "triad"], default="triad", help="Select the STREAM operation")
    parser.add_argument("--mode", choices=["vv", "vf"], default="vf", help="Select the requested instruction format (vv, vf, default is vf)")
    parser.add_argument("--lmul", type=int, choices=[1, 2, 4, 8], default=1, help="LMUL (1, 2, 4, 8) - only applies to RVV")
    parser.add_argument("--unroll", type=int, default=1, help="Maximum manual unrolling factor")
    args = parser.parse_args()

    dt = adt.FP64
    isa = args.isa

    # Dynamically load the correct ISA generator
    module_name = isa_modules.get(isa, isa)
    try:
        generator_module = importlib.import_module(f"asmgen.asmblocks.{module_name}")
        generator_class = getattr(generator_module, isa)
    except (ImportError, AttributeError):
        print(f"Error: Could not load ISA module for '{isa}'")
        sys.exit(1)

    gen: asmgen = generator_class()
    gen.set_output_inline(yesno=False)

    if isa == "rvv":
        gen.set_parameter("LMUL", args.lmul)

    rt = reg_tracker(reg_type_init_list=[
        ("greg", gen.max_gregs),
        ("freg", gen.max_fregs),
        ("vreg", gen.max_vregs),
        ("treg", gen.max_tregs(dt=dt)),
    ])
    cc = gen.create_callconv()
    func = fngen(gen=gen, rt=rt)

    # Function mapping
    cc.add_param("greg", "n")
    cc.add_param("freg", "alpha", dt.FP64)
    cc.add_param("greg", "a")
    cc.add_param("greg", "b")
    cc.add_param("greg", "c")

    op_names = {
        "copy": "asm_Copy",
        "scale": "asm_Scale",
        "add": "asm_Add",
        "triad": "asm_Triad"
    }
    fnname = op_names.get(args.op)

    func.init_cc(cc=cc)

    n_addr_idx = rt.aliased_regs["greg"]["n"]
    alpha_idx = rt.aliased_regs["freg"]["alpha"]
    a_addr_idx = rt.aliased_regs["greg"]["a"]
    b_addr_idx = rt.aliased_regs["greg"]["b"]
    c_addr_idx = rt.aliased_regs["greg"]["c"]
    
    asmheader = (
        ".section .text\n"
        f".global {fnname}\n"
        f"{fnname}:\n  "
    )

    n_addr = gen.greg(n_addr_idx)
    alpha_scalar = gen.freg(alpha_idx, dt=dt)
    a_addr = gen.greg(a_addr_idx)
    b_addr = gen.greg(b_addr_idx)
    c_addr = gen.greg(c_addr_idx)

    # Call by reference for the dimension as the tail handling is done externally
    n_idx = rt.reserve_any_reg("greg")
    n = gen.greg(n_idx)
    
    # Check for Vector-Float (vf) capability dynamically based on the active ISA module
    can_vf = True
    fma_mods = {mod.VF}
    if args.mode == "vf":
        try:
            # Dummy test instruction to check if `vf` modifiers throw an exception
            gen.fma(
                adreg=gen.vreg(0), bdreg=gen.freg(1, dt), cdreg=gen.vreg(2),
                a_dt=dt, b_dt=dt, c_dt=dt, modifiers=fma_mods
            )
        except Exception:
            can_vf = False
            fma_mods = set()
    else:
        can_vf = False
        fma_mods = set()

    # If vf is unsupported or vv is requested, reserve a broadcast vector for alpha
    alpha_vector = None
    if not can_vf and args.op in ["scale", "triad"]:
        alpha_vector_idx = rt.reserve_any_reg("vreg")
        alpha_vector = gen.vreg(alpha_vector_idx)

    asmblock = gen.isaquirks(dt=dt, rt=rt)

    asmblock += gen.load_greg(areg=n_addr, offset=0, dst=n)

    # Determine if we are utilizing a Variable Length Agnostic (VLA) ISA like RVV
    is_vla = "vlen" in rt.aliased_regs["greg"]
    vlen = None
    vlenb = None

    if is_vla:
        vlen_idx = rt.aliased_regs["greg"]["vlen"]
        vlen = gen.greg(vlen_idx)
        vlenb_idx = rt.reserve_any_reg("greg")
        vlenb = gen.greg(vlenb_idx)

        asmblock += gen.mov_greg(src=vlen, dst=vlenb)
        asmblock += gen.shift_greg_left(reg=vlenb, bit_count=adt_size(dt).bit_length()-1)
    else:
        vlen_idx = rt.reserve_any_reg("greg")
        vlen = gen.greg(vlen_idx)

        asmblock += gen.simd_size_to_greg(reg=vlen, dt=dt)

    # Broadcast scalar freg to vreg if vf is unsupported/unwanted
    if not can_vf and args.op in ["scale", "triad"]:
            asmblock += gen.fill_vector(sreg=alpha_scalar, vreg=alpha_vector, dt=dt)

    # Calculate maximum unroll factor
    regs_per_group = 1 if args.op in ["copy", "scale"] else 2
    regs_used_by_scalar = 1 if (not can_vf and args.op in ["scale", "triad"]) else 0
    unroll_factor = (gen.max_vregs - regs_used_by_scalar) // regs_per_group
    unroll_factor = min(args.unroll, unroll_factor)
    
    n_block_idx = rt.reserve_any_reg("greg")
    n_block = gen.greg(n_block_idx)

    asmblock += gen.mul_greg_imm(src=vlen, dst=n_block, factor=unroll_factor)

    n_thresh_idx = rt.reserve_any_reg("greg")
    n_thresh = gen.greg(n_thresh_idx)
    asmblock += gen.mul_greg_imm(src=n_block, dst=n_thresh, factor=1)
    
    vreg_groups = []
    for _ in range(unroll_factor):
        v1_idx = rt.reserve_any_reg("vreg")
        v1 = gen.vreg(v1_idx)
        if args.op in ["add", "triad"]:
            v2_idx = rt.reserve_any_reg("vreg")
            v2 = gen.vreg(v2_idx)
            vreg_groups.append((v1, v2))
        else:
            vreg_groups.append((v1, None))

    # Helper function for generic pointer arithmetic
    def advance_ptr(dst_reg):
        if is_vla:
            return gen.add_greg_greg(dst=dst_reg, reg1=dst_reg, reg2=vlenb)
        else:
            return gen.add_greg_voff(reg=dst_reg, offset=1, dt=dt)

    # Exit if size is too small
    asmblock += gen.cb(reg1=n, reg2=n_thresh, cmp=comparison.LT, label="goodbye")
        
    # Main unrolled loop
    asmblock += gen.label(label="loop")

    # Load block
    for v1, v2 in vreg_groups:
        if args.op == "copy":
            asmblock += gen.load_vector(areg=a_addr, vreg=v1, dt=dt)
            asmblock += advance_ptr(a_addr)
        elif args.op == "scale":
            asmblock += gen.load_vector(areg=c_addr, vreg=v1, dt=dt)
            asmblock += advance_ptr(c_addr)
        elif args.op == "add":
            asmblock += gen.load_vector(areg=a_addr, vreg=v1, dt=dt)
            asmblock += advance_ptr(a_addr)
            asmblock += gen.load_vector(areg=b_addr, vreg=v2, dt=dt)
            asmblock += advance_ptr(b_addr)
        elif args.op == "triad":
            asmblock += gen.load_vector(areg=b_addr, vreg=v1, dt=dt)
            asmblock += advance_ptr(b_addr)
            asmblock += gen.load_vector(areg=c_addr, vreg=v2, dt=dt)
            asmblock += advance_ptr(c_addr)

    # Compute block
    for v1, v2 in vreg_groups:
        if args.op == "scale":
            breg = alpha_scalar if can_vf else alpha_vector
            asmblock += gen.fmul(adreg=v1, bdreg=breg, cdreg=v1,
                                 a_dt=dt, b_dt=dt, c_dt=dt, modifiers=fma_mods)
        elif args.op == "add":
            asmblock += gen.fadd(adreg=v1, bdreg=v2, cdreg=v1,
                                 a_dt=dt, b_dt=dt, c_dt=dt)
        elif args.op == "triad":
            breg = alpha_scalar if can_vf else alpha_vector
            asmblock += gen.fma(adreg=v2, bdreg=breg, cdreg=v1,
                                a_dt=dt, b_dt=dt, c_dt=dt, modifiers=fma_mods)

    # Store block
    for v1, v2 in vreg_groups:
        if args.op == "copy":
            asmblock += gen.store_vector(areg=c_addr, vreg=v1, dt=dt)
            asmblock += advance_ptr(c_addr)
        elif args.op == "scale":
            asmblock += gen.store_vector(areg=b_addr, vreg=v1, dt=dt)
            asmblock += advance_ptr(b_addr)
        elif args.op == "add":
            asmblock += gen.store_vector(areg=c_addr, vreg=v1, dt=dt)
            asmblock += advance_ptr(c_addr)
        elif args.op == "triad":
            asmblock += gen.store_vector(areg=a_addr, vreg=v1, dt=dt)
            asmblock += advance_ptr(a_addr)

    # Loop epilogue
    asmblock += gen.sub_greg_greg(dst=n, reg1=n, reg2=n_block)
    asmblock += gen.cb(reg1=n, reg2=n_thresh, cmp=comparison.GE, label="loop")

    # Skip tail if possible
    #asmblock += gen.cb(reg1=n, reg2=None, cmp=comparison.NZ, label="goodbye")
    
    # Tail loop (Only applicable for ISAs supporting dynamic length like vsetvli)
    if is_vla and hasattr(gen, "vsetvli"):
        asmblock += gen.label(label="tail")
        asmblock += gen.vsetvli(vlreg=vlen, avlreg=n, dt=dt)
        asmblock += gen.mov_greg(src=vlen, dst=vlenb)
        asmblock += gen.shift_greg_left(reg=vlenb, bit_count=adt_size(dt).bit_length()-1)
        
        v1, v2 = vreg_groups[0]
        
        # Tail load
        if args.op == "copy":
            asmblock += gen.load_vector(areg=a_addr, vreg=v1, dt=dt)
            asmblock += advance_ptr(a_addr)
        elif args.op == "scale":
            asmblock += gen.load_vector(areg=c_addr, vreg=v1, dt=dt)
            asmblock += advance_ptr(c_addr)
        elif args.op == "add":
            asmblock += gen.load_vector(areg=a_addr, vreg=v1, dt=dt)
            asmblock += advance_ptr(a_addr)
            asmblock += gen.load_vector(areg=b_addr, vreg=v2, dt=dt)
            asmblock += advance_ptr(b_addr)
        elif args.op == "triad":
            asmblock += gen.load_vector(areg=b_addr, vreg=v1, dt=dt)
            asmblock += advance_ptr(b_addr)
            asmblock += gen.load_vector(areg=c_addr, vreg=v2, dt=dt)
            asmblock += advance_ptr(c_addr)

        # Tail compute
        if args.op == "scale":
            breg = alpha_scalar if can_vf else alpha_vector
            asmblock += gen.fmul(adreg=v1, bdreg=breg, cdreg=v1,
                                 a_dt=dt, b_dt=dt, c_dt=dt, modifiers=fma_mods)
        elif args.op == "add":
            asmblock += gen.fadd(adreg=v1, bdreg=v2, cdreg=v1,
                                 a_dt=dt, b_dt=dt, c_dt=dt)
        elif args.op == "triad":
            breg = alpha_scalar if can_vf else alpha_vector
            asmblock += gen.fma(adreg=v2, bdreg=breg, cdreg=v1,
                                a_dt=dt, b_dt=dt, c_dt=dt, modifiers=fma_mods)

        # Tail store
        if args.op == "copy":
            asmblock += gen.store_vector(areg=c_addr, vreg=v1, dt=dt)
            asmblock += advance_ptr(c_addr)
        elif args.op == "scale":
            asmblock += gen.store_vector(areg=b_addr, vreg=v1, dt=dt)
            asmblock += advance_ptr(b_addr)
        elif args.op == "add":
            asmblock += gen.store_vector(areg=c_addr, vreg=v1, dt=dt)
            asmblock += advance_ptr(c_addr)
        elif args.op == "triad":
            asmblock += gen.store_vector(areg=a_addr, vreg=v1, dt=dt)
            asmblock += advance_ptr(a_addr)

        asmblock += gen.sub_greg_greg(dst=n, reg1=n, reg2=vlen)
        asmblock += gen.cb(reg1=n, reg2=None, cmp=comparison.NZ, label="tail")

    # Finalize
    asmblock += gen.label(label="goodbye")

    # Store final remainder
    asmblock += gen.store_greg(areg=n_addr, offset=0, src=n)
    
    asmblock += gen.isaendquirks(dt=dt, rt=rt)

    fnsave, fnload, fnrestore = func.get_boilerplate(cc=cc)
    
    print(asmheader + fnsave + fnload + asmblock + fnrestore + "ret")

if __name__ == "__main__":
    main()
    
