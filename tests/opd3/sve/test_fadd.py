# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Tests SVE fadd instruction code generation
"""

from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.op import opd3_modifier as mod

from .test_sve_opd3 import test_sve_opd3_base

class test_sve_fadd(test_sve_opd3_base):
    """
    Testsuite for SVE fadd instruction
    """

    def test_standard_fadd(self):
        """
        Test some known instructions for correctness
        """
        cases = [
            (adt.FP64, "fadd z0.d,p0/m,z0.d,z2.d\n"),
            (adt.FP32, "fadd z0.s,p0/m,z0.s,z2.s\n"),
            (adt.SINT32, "add z0.s,p0/m,z0.s,z2.s\n")
        ]
        for dt, expected in cases:
            with self.subTest(dt=dt.name):
                res = self.gen.fadd(adreg=self.v0, bdreg=self.v2, cdreg=self.v0, amreg=self.p0,
                                    a_dt=dt, b_dt=dt, c_dt=dt, modifiers={mod.MASK})
                self.assertEqual(res, expected)

    def test_invalid_configurations(self):
        """
        Test that using invalid configurations results in the expected error being raised
        """
        with self.subTest(error="unsupported NP"):
            with self.assertRaisesRegex(ValueError, "add has no NP-form"):
                self.gen.fadd(adreg=self.v0, bdreg=self.v2, cdreg=self.v0, amreg=self.p0,
                              a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32,
                              modifiers={mod.MASK, mod.NP})
