# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Tests NEON/ASIMD fmul instruction code generation
"""

from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.op import opd3_modifier as mod

from .test_neon_opd3 import test_neon_opd3_base

class test_neon_mul(test_neon_opd3_base):
    """
    Testsuite for NEON/ASIMD fmul
    """

    def test_standard_mul(self):
        """
        Test some known instructions for correctness
        """
        cases = [
            (adt.FP64, "fmul v0.2d,v1.2d,v2.2d\n"),
            (adt.FP32, "fmul v0.4s,v1.4s,v2.4s\n"),
            (adt.FP16, "fmul v0.8h,v1.8h,v2.8h\n"),
            (adt.SINT16, "mul v0.8h,v1.8h,v2.8h\n")
        ]
        for dt, expected in cases:
            with self.subTest(dt=dt.name):
                res = self.gen.mul(adreg=self.v1, bdreg=self.v2, cdreg=self.v0,
                                    a_dt=dt, b_dt=dt, c_dt=dt)
                self.assertEqual(res, expected)

    def test_invalid_configurations(self):
        """
        Test that invalid configurations raise the correct Error
        """
        # NP is unsupported in fmul
        with self.subTest(error="unsupported NP"):
            with self.assertRaisesRegex(ValueError, "NEON mul has no NP-form"):
                self.gen.mul(adreg=self.v1, bdreg=self.v2, cdreg=self.v0,
                              a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32,
                              modifiers={mod.NP})

        # Widening without PART modifier
        with self.subTest(error="widening missing PART"):
            with self.assertRaisesRegex(ValueError, "Invalid configuration for neon_mul"):
                self.gen.mul(adreg=self.v1, bdreg=self.v2, cdreg=self.v0,
                              a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP32)

        # Non-widening integer MUL
        with self.subTest(error="unsupported NP"):
            with self.assertRaisesRegex(
                    ValueError,
                    "Only widening unsigned integer MUL supported in NEON"):
                self.gen.mul(adreg=self.v1, bdreg=self.v2, cdreg=self.v0,
                              a_dt=adt.UINT32, b_dt=adt.UINT32, c_dt=adt.UINT32,
                              modifiers=set())
