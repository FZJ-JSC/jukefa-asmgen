# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Test Base X86_64 loads/stores
"""
import unittest

from asmgen.asmblocks.avx.types import avx_freg, reg_prefixer, x86_greg
from asmgen.asmblocks.op import operand_modifier as opd_mod
from asmgen.asmblocks.op import opmem_modifier as mod
from asmgen.asmblocks.x86.opmem import x86_load, x86_store
from asmgen.registers import asm_data_type as adt


class test_x86_opmem(unittest.TestCase):
    """
    Testsuite for X86_64 opmem instructions
    """
    def setUp(self):
        self.r8 = x86_greg(0)
        self.r15 = x86_greg(7)

        self.xmm0_32 = avx_freg(0)
        self.xmm1_64 = avx_freg(1)
        self.xmm2_16 = avx_freg(2)

        self.load = x86_load(asmwrap = lambda s : f"{s}\n",
                             rpref=reg_prefixer(lambda: False))
        self.store = x86_store(asmwrap = lambda s : f"{s}\n",
                               rpref=reg_prefixer(lambda: False))

    def test_basic_integer_loads(self):
        """ Test scalar integer zero-offset loads across sizes """
        self.assertEqual(
            self.load(dregs=[self.r8], areg=self.r15, dt=adt.UINT8, modifiers=set()),
            "movb (%r15), %r8\n"
        )
        self.assertEqual(
            self.load(dregs=[self.r8], areg=self.r15, dt=adt.UINT16, modifiers=set()),
            "movw (%r15), %r8\n"
        )
        self.assertEqual(
            self.load(dregs=[self.r8], areg=self.r15, dt=adt.UINT32, modifiers=set()),
            "movl (%r15), %r8\n"
        )
        self.assertEqual(
            self.load(dregs=[self.r8], areg=self.r15, dt=adt.UINT64, modifiers=set()),
            "movq (%r15), %r8\n"
        )

    def test_basic_float_loads(self):
        """ Test scalar floating point zero-offset loads """
        self.assertEqual(
            self.load(dregs=[self.xmm2_16], areg=self.r15, dt=adt.FP16, modifiers=set()),
            "vmovsh (%r15), %xmm2\n"
        )
        self.assertEqual(
            self.load(dregs=[self.xmm0_32], areg=self.r15, dt=adt.FP32, modifiers=set()),
            "vmovss (%r15), %xmm0\n"
        )
        self.assertEqual(
            self.load(dregs=[self.xmm1_64], areg=self.r15, dt=adt.FP64, modifiers=set()),
            "vmovsd (%r15), %xmm1\n"
        )

    def test_immediate_offset_stores(self):
        """ Test scalar stores with immediate offsets """
        self.assertEqual(
            self.store(dregs=[self.r8], areg=self.r15, dt=adt.UINT64,
                       modifiers={mod.IOFFSET}, ioffset=32),
            "movq %r8, 32(%r15)\n"
        )
        self.assertEqual(
            self.store(dregs=[self.xmm1_64], areg=self.r15, dt=adt.FP64,
                       modifiers={mod.IOFFSET}, ioffset=-16),
            "vmovsd %xmm1, -16(%r15)\n"
        )

    def test_invalid_modifiers(self):
        """ Test that invalid modifiers are properly rejected """
        invalid_mods = [mod.TINDEX, mod.VINDEX, mod.POSTINC,
                        mod.TOFFSET, mod.VOFFSET, mod.ISTRIDE, mod.GSTRIDE,
                        mod.STRUCT, mod.MASK]

        # none are supported
        invalid_opd_mods = list(opd_mod)

        for invalid_mod in invalid_mods:
            for invalid_opd_mod in invalid_opd_mods:
                with self.subTest(modifier=invalid_mod):
                    with self.assertRaisesRegex(ValueError, "Base X86_64 has no"):
                        self.load(dregs=[self.r8], areg=self.r15,
                                  dt=adt.UINT64,
                                  modifiers={invalid_mod},
                                  operand_modifiers={'adreg':{invalid_opd_mod}})

    def test_missing_ioffset(self):
        """ Test kwargs validation for IOFFSET """
        with self.assertRaisesRegex(ValueError, "IOFFSET modifier requires 'ioffset' parameter"):
            self.load(dregs=[self.r8], areg=self.r15, dt=adt.UINT64, modifiers={mod.IOFFSET})


if __name__ == '__main__':
    unittest.main()
