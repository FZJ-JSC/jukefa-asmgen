# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Tests SVE fmla instruction code generation
"""

from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.op import optransform_modifier as mod
from asmgen.asmblocks.op import operand_modifier as opd_mod
from asmgen.asmblocks.op.optransform import widening_method as wm

from .test_sve_opd3 import test_sve_opd3_base

class test_sve_fmla(test_sve_opd3_base):
    """
    Testsuite for SVE fmla and similar instructions
    """

    def test_standard_fmla(self):
        """
        Test valid standard MASKED FMLA operations across data types
        """
        cases = [
            (adt.FP64, "fmla z0.d,p0/m,z1.d,z2.d\n"),
            (adt.FP32, "fmla z0.s,p0/m,z1.s,z2.s\n"),
            (adt.FP16, "fmla z0.h,p0/m,z1.h,z2.h\n"),
            (adt.SINT16, "mla z0.h,p0/m,z1.h,z2.h\n") # Ints drop the 'f' prefix
        ]
        for dt, expected in cases:
            with self.subTest(dt=dt.name):
                res = self.gen.fma(adreg=self.v1, bdreg=self.v2, cdreg=self.v0, amreg=self.p0,
                                   a_dt=dt, b_dt=dt, c_dt=dt, modifiers={mod.MASK})
                self.assertEqual(res, expected)

    def test_widening_and_np(self):
        """
        Test unpredicated widening (PART) and negate-product (NP) combinations
        """
        cases = [
            # 2x Widening
            (adt.FP16, adt.FP32, 0, set(), "fmlalb z0.s,z1.h,z2.h\n"),
            (adt.FP16, adt.FP32, 1, set(), "fmlalt z0.s,z1.h,z2.h\n"),
            # 4x Widening + Negate Product
            (adt.FP8E5M2, adt.FP32, 0, {mod.NP}, "fmlsllbb z0.s,z1.b,z2.b\n"),
            (adt.FP8E5M2, adt.FP32, 3, {mod.NP}, "fmlslltt z0.s,z1.b,z2.b\n"),
        ]
        for narrow_dt, wide_dt, part, mods, expected in cases:
            modifiers = mods | {mod.PART}
            with self.subTest(narrow=narrow_dt.name, wide=wide_dt.name, part=part, mods=mods):
                res = self.gen.fma(adreg=self.v1, bdreg=self.v2, cdreg=self.v0,
                                   a_dt=narrow_dt, b_dt=narrow_dt, c_dt=wide_dt,
                                   modifiers=modifiers, part=part,
                                   widening_method=wm.SPLIT_INTERLEAVED)
                self.assertEqual(res, expected)

    def test_indexed_fma(self):
        """
        Test unpredicated indexed operations (BLOCKLANE)
        """
        # FP32 allows idx 0-3 and bdreg z0-z7
        res = self.gen.fma(adreg=self.v1, bdreg=self.v7, cdreg=self.v0,
                           a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32,
                           operand_modifiers={ 'bdreg' : {opd_mod.BLOCKLANE}},
                           bdreg_blocksize=4, bdreg_lane=3)
        self.assertEqual(res, "fmla z0.s,z1.s,z7.s[3]\n")

        # FP64 allows idx 0-1 and bdreg z0-z15
        res = self.gen.fma(adreg=self.v1, bdreg=self.v15, cdreg=self.v0,
                           a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64,
                           operand_modifiers={ 'bdreg' : {opd_mod.BLOCKLANE}},
                           bdreg_blocksize=2, bdreg_lane=1)
        self.assertEqual(res, "fmla z0.d,z1.d,z15.d[1]\n")

    def test_constraint_boundaries(self):
        """
        Test that structural constraints catch architectural violations
        """

        # 1. BLOCKLANE index out of bounds (FP32 blocksize=4, max index=3)
        with self.subTest(error="blockidx index bounds"):
            with self.assertRaisesRegex(ValueError,
                                        "value of bdreg_lane must be <= 3"):
                self.gen.fma(adreg=self.v1, bdreg=self.v2, cdreg=self.v0,
                             a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32,
                             operand_modifiers={ 'bdreg' : {opd_mod.BLOCKLANE}},
                             bdreg_blocksize=4, bdreg_lane=4)

        # 2. BLOCKLANE register out of bounds (FP32 restricts Zm to z0-z7)
        with self.subTest(error="blockidx bdreg bounds FP32"):
            with self.assertRaisesRegex(ValueError, "index of bdreg must be <= 7"):
                self.gen.fma(adreg=self.v1, bdreg=self.v8, cdreg=self.v0,
                             a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32,
                             operand_modifiers={ 'bdreg' : {opd_mod.BLOCKLANE}},
                             bdreg_blocksize=4, bdreg_lane=1)

        # 3. BLOCKLANE register out of bounds (FP64 restricts Zm to z0-z15)
        with self.subTest(error="blockidx bdreg bounds FP64"):
            with self.assertRaisesRegex(ValueError, "index of bdreg must be <= 15"):
                self.gen.fma(adreg=self.v1, bdreg=self.v16, cdreg=self.v0,
                             a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64,
                             operand_modifiers={ 'bdreg' : {opd_mod.BLOCKLANE}},
                             bdreg_blocksize=2, bdreg_lane=0)

        # 4. PART out of bounds (FP16 -> FP32 max part is 1)
        with self.subTest(error="part bounds"):
            with self.assertRaisesRegex(ValueError, "value of part must be <= 1"):
                self.gen.fma(adreg=self.v1, bdreg=self.v2, cdreg=self.v0,
                             a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP32,
                             modifiers={mod.PART}, part=2, widening_method=wm.SPLIT_INTERLEAVED)

    def test_invalid_configurations(self):
        """
        Test that using invalid configurations results in the expected error being raised
        """
        # 1. Missing amreg when MASK is active
        with self.subTest(error="missing mask reg"):
            with self.assertRaisesRegex(ValueError, "Invalid configuration for sve_fma"):
                self.gen.fma(adreg=self.v1, bdreg=self.v2, cdreg=self.v0,
                             a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32,
                             modifiers={mod.MASK}) # amreg=self.p0 missing!

        # 2. Unsigned non-widening (caught by shared diagnose_failure hook)
        with self.subTest(error="unsigned non-widening"):
            with self.assertRaisesRegex(ValueError, "Only widening unsigned integer ML supported"):
                self.gen.fma(adreg=self.v1, bdreg=self.v2, cdreg=self.v0, amreg=self.p0,
                             a_dt=adt.UINT32, b_dt=adt.UINT32, c_dt=adt.UINT32,
                             modifiers={mod.MASK})
