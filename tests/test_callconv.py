# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Test the Calling convention abstraction
"""

import unittest


from parameterized import parameterized_class
from asmgen.asmblocks.noarch import asmgen
from asmgen.asmblocks.avx import fma128,fma256,avx512
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.rvv071 import rvv071
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.sme import sme

# base ISA classes
from asmgen.asmblocks.riscv64 import riscv64
from asmgen.asmblocks.aarch64 import aarch64
from asmgen.asmblocks.avx.base import avxbase

class callconv_testsuite_type_hinter:
    """
    Type hinting helper for callconv test cases
    """
    gen : asmgen
    name : str

@parameterized_class([
    {"name": "fma128", "gen": fma128()},
    {"name": "fma256", "gen": fma256()},
    {"name": "avx512", "gen": avx512()},
    {"name": "neon", "gen": neon()},
    {"name": "sve", "gen": sve()},
    {"name": "rvv", "gen": rvv()},
    {"name": "rvv071", "gen": rvv071()},
])
class test_callconv_gemmukr(unittest.TestCase,callconv_testsuite_type_hinter):
    """
    Testsuite for calling conventions
    """

    def setUp(self):
        self.cc = self.gen.create_callconv()


        self.cc.add_param('greg', "m")
        self.cc.add_param('greg', "n")
        self.cc.add_param('greg', "k")
        self.cc.add_param('greg', "alpha")
        self.cc.add_param('greg', "a")
        self.cc.add_param('greg', "b")
        self.cc.add_param('greg', "beta")
        self.cc.add_param('greg', "c")
        self.cc.add_param('greg', "rs_c")
        self.cc.add_param('greg', "cs_c")
        self.cc.add_param('greg', "data")
        self.cc.add_param('greg', "cntx")


        self.gregs_used = []
        self.fregs_used = []

        for name,(tag,idx,in_stack,dt) in self.cc.get_params().items():
            del name, dt

            if not in_stack:
                if 'greg' == tag:
                    self.gregs_used.append(idx)
                elif 'freg' == tag:
                    self.fregs_used.append(idx)


    expected_save_blocks = [
        (riscv64,(
            "add sp,sp,-64\n"
            "sd a0,0(sp)\n"
            "sd a1,8(sp)\n"
            "sd a2,16(sp)\n"
            "sd a3,24(sp)\n"
            "sd a4,32(sp)\n"
            "sd a5,40(sp)\n"
            "sd a6,48(sp)\n"
            "sd a7,56(sp)\n"
        )),
        (aarch64, (
            "add sp,sp,#-64\n"
            "str x0,[sp,#0]\n"
            "str x1,[sp,#8]\n"
            "str x2,[sp,#16]\n"
            "str x3,[sp,#24]\n"
            "str x4,[sp,#32]\n"
            "str x5,[sp,#40]\n"
            "str x6,[sp,#48]\n"
            "str x7,[sp,#56]\n")
         ),
        (avxbase,(
            "addq $-48,%rsp\n"
            "movq %r8,(%rsp)\n"
            "movq %r9,8(%rsp)\n"
            "movq %rcx,16(%rsp)\n"
            "movq %rdx,24(%rsp)\n"
            "movq %rsi,32(%rsp)\n"
            "movq %rdi,40(%rsp)\n")
         )
    ]

    expected_restore_blocks = [
        (riscv64,(
            "ld a0,0(sp)\n"
            "ld a1,8(sp)\n"
            "ld a2,16(sp)\n"
            "ld a3,24(sp)\n"
            "ld a4,32(sp)\n"
            "ld a5,40(sp)\n"
            "ld a6,48(sp)\n"
            "ld a7,56(sp)\n"
            "add sp,sp,64\n")
         ),
        (aarch64, (
            "ldr x0,[sp,#0]\n"
            "ldr x1,[sp,#8]\n"
            "ldr x2,[sp,#16]\n"
            "ldr x3,[sp,#24]\n"
            "ldr x4,[sp,#32]\n"
            "ldr x5,[sp,#40]\n"
            "ldr x6,[sp,#48]\n"
            "ldr x7,[sp,#56]\n"
            "add sp,sp,#64\n")
         ),
        (avxbase,(
            "movq (%rsp),%r8\n"
            "movq 8(%rsp),%r9\n"
            "movq 16(%rsp),%rcx\n"
            "movq 24(%rsp),%rdx\n"
            "movq 32(%rsp),%rsi\n"
            "movq 40(%rsp),%rdi\n"
            "addq $48,%rsp\n")
         )
    ]

    def test_save_restore(self):
        """
        Test that save and restore blocks are correct
        """

        expected_save_block = "UNKNOWN"
        for base,block in self.expected_save_blocks:
            if isinstance(self.gen,base):
                expected_save_block = block

        expected_restore_block = "UNKNOWN"
        for base,block in self.expected_restore_blocks:
            if isinstance(self.gen,base):
                expected_restore_block = block

        self.gen.set_output_inline(yesno=False)

        save_block = self.cc.save_before_call(
                gen=self.gen,
                regs={
                    'greg': self.gregs_used,
                    'freg': self.fregs_used,
                })
        restore_block = self.cc.restore_after_call(
                gen=self.gen,
                regs={
                    'greg': self.gregs_used,
                    'freg': self.fregs_used,
                })


        self.assertEqual(expected_save_block, save_block)
        self.assertEqual(expected_restore_block, restore_block)


class test_clobbers(unittest.TestCase):
    """
    Registers clobbered beyond the allocation, kept by the calling
    convention where they are the callee's
    """

    def test_callconv_keeps_callee_saved_clobbers(self):
        g = sme()
        g.output_inline = False
        cc = g.create_callconv()
        cc.add_clobber("freg", range(32))
        save = cc.save_in_call(g, regs={"greg": [], "freg": []})
        # AAPCS64: d8-d15 the callee's, the others not
        self.assertEqual(sorted(f"d{n}" for n in range(8, 16)),
                         sorted(l.split()[1].split(",")[0] for l in save.splitlines()
                                if l.strip().startswith("str")))


class test_callee_saved_last(unittest.TestCase):
    """
    The calling convention's callee-saved registers allocated last - through
    the tracker's shared pools for vector registers whose low parts they are
    """

    def test_deprioritized(self):
        g = sme()
        cc = g.create_callconv()
        rt = g.get_reg_tracker()
        cc.deprioritize_callee_saved(rt)
        # AAPCS64: x19-x28 and d8-d15 - so v8-v15, their registers - last
        self.assertEqual(-1, rt.priority("greg", 19))
        self.assertEqual(-1, rt.priority("vreg", 8))
        free = [i for i in range(g.max_vregs()) if i not in rt.get_used_regs("vreg")]
        order = [rt.reserve_any_reg("vreg") for _ in range(len(free) - 8)]
        self.assertFalse(set(order) & set(range(8, 16)))


class test_stack_parameters(unittest.TestCase):
    """
    Parameters passed on the stack are read past the save area the
    convention lays out - its clobbers included
    """

    def test_offset_past_saves(self):
        from asmgen.callconv.fngen import fngen
        from asmgen.registers import asm_data_type as adt
        g = sme()
        g.output_inline = False
        rt = g.get_reg_tracker()
        cc = g.create_callconv()
        fn = fngen(gen=g, rt=rt)
        # nine parameters: the ninth on the stack
        for i in range(9):
            cc.add_param("greg", f"p{i}")
        fn.init_cc(cc=cc)
        cc.add_clobber("freg", range(32))
        save, load, restore = fn.get_boilerplate(cc=cc)
        if "ldr" not in load:
            self.skipTest("no parameter on the stack in this setup")
        self.assertIn("[sp,#64]", load.replace(" ", ""))
