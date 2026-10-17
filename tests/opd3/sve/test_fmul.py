# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Tests SVE fmul instruction code generation
"""

from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.op import opd3_modifier as mod
from asmgen.asmblocks.op.opd3 import widening_method as wm

from .test_sve_opd3 import test_sve_opd3_base

class test_sve_fmul(test_sve_opd3_base):
    """
    Testsuite for SVE fmla and similar instructions
    """

    def test_standard_fmul(self):
        """
        Test some known instructions for correctness
        """
        cases = [
            (adt.FP64, "fmul z0.d,p0/m,z0.d,z2.d\n"),
            (adt.FP32, "fmul z0.s,p0/m,z0.s,z2.s\n"),
            (adt.SINT16, "mul z0.h,p0/m,z0.h,z2.h\n")
        ]
        for dt, expected in cases:
            with self.subTest(dt=dt.name):
                res = self.gen.fmul(adreg=self.v0, bdreg=self.v2, cdreg=self.v0, amreg=self.p0,
                                    a_dt=dt, b_dt=dt, c_dt=dt, modifiers={mod.MASK})
                self.assertEqual(res, expected)

    def test_invalid_configurations(self):
        """
        Test that using invalid configurations results in the expected error being raised
        """
        # 1. NP is unsupported in FMUL
        with self.subTest(error="unsupported NP"):
            with self.assertRaisesRegex(ValueError, "SVE mul has no NP-form"):
                self.gen.fmul(adreg=self.v0, bdreg=self.v2, cdreg=self.v0, amreg=self.p0,
                              a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32,
                              modifiers={mod.MASK, mod.NP})

        # 2. Widening without PART
        with self.subTest(error="widening missing PART"):
            with self.assertRaisesRegex(ValueError, "Invalid configuration for sve_fmul"):
                self.gen.fmul(adreg=self.v1, bdreg=self.v2, cdreg=self.v0,
                              a_dt=adt.SINT16, b_dt=adt.SINT16, c_dt=adt.SINT32,
                              widening_method=wm.SPLIT_INSTRUCTIONS) # modifiers={mod.PART} missing
