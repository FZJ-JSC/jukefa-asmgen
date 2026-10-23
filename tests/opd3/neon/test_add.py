# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Tests NEON/ASIMD fadd instruction code generation
"""
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.op import opd3_modifier as mod

from .test_neon_opd3 import test_neon_opd3_base

class test_neon_add(test_neon_opd3_base):
    """
    Testsuite for NEON/ASIMD fadd
    """

    def test_standard_add(self):
        """
        Correctness tests for some known cases
        """
        cases = [
            (adt.FP64, "fadd v0.2d,v1.2d,v2.2d\n"),
            (adt.FP32, "fadd v0.4s,v1.4s,v2.4s\n"),
            (adt.SINT32, "add v0.4s,v1.4s,v2.4s\n")
        ]
        for dt, expected in cases:
            with self.subTest(dt=dt.name):
                res = self.gen.add(adreg=self.v1, bdreg=self.v2, cdreg=self.v0,
                                    a_dt=dt, b_dt=dt, c_dt=dt)
                self.assertEqual(res, expected)

    def test_invalid_configurations(self):
        """
        Test that invalid configurations raise the expected errors
        """
        # 1. NP is unsupported in fadd
        with self.subTest(error="unsupported NP"):
            with self.assertRaisesRegex(ValueError, "NEON add has no NP-form"):
                self.gen.add(adreg=self.v1, bdreg=self.v2, cdreg=self.v0,
                              a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32,
                              modifiers={mod.NP})
