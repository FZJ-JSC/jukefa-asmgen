# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Tests NEON/ASIMD fma instruction code generation
"""

from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.op import (
    opd3_modifier as mod,
    operand_modifier as opd_mod
)

from asmgen.asmblocks.op.opd3 import widening_method as wm

from .test_neon_opd3 import test_neon_opd3_base

class test_neon_fmla(test_neon_opd3_base):
    """
    Testsuite for NEON/ASIMD fma
    """

    def test_standard_fmla(self):
        """Test valid standard FMLA operations across data types"""
        cases = [
            (adt.FP64, "fmla v0.2d,v1.2d,v2.2d\n"),
            (adt.FP32, "fmla v0.4s,v1.4s,v2.4s\n"),
            (adt.FP16, "fmla v0.8h,v1.8h,v2.8h\n"),
            (adt.SINT16, "mla v0.8h,v1.8h,v2.8h\n")
        ]
        for dt, expected in cases:
            with self.subTest(dt=dt.name):
                res = self.gen.fma(adreg=self.v1, bdreg=self.v2, cdreg=self.v0,
                                   a_dt=dt, b_dt=dt, c_dt=dt)
                self.assertEqual(res, expected)

    def test_widening_and_np(self):
        """Test widening (PART) and negate-product (NP) combinations"""
        cases = [
            # 2x Widening
            # FP16 to FP32: the lower half (fmlal), the upper half (fmlal2)
            (adt.FP16, adt.FP32, 0, set(), "fmlal v0.4s,v1.4h,v2.4h\n"),
            (adt.FP16, adt.FP32, 1, set(), "fmlal2 v0.4s,v1.4h,v2.4h\n"),
            # 4x Widening + Negate Product
            (adt.FP8E5M2, adt.FP32, 0, {mod.NP}, "fmlsllbb v0.4s,v1.16b,v2.16b\n"),
            (adt.FP8E5M2, adt.FP32, 3, {mod.NP}, "fmlslltt v0.4s,v1.16b,v2.16b\n"),
        ]
        for narrow_dt, wide_dt, part, mods, expected in cases:
            modifiers = mods | {mod.PART}
            with self.subTest(narrow=narrow_dt.name, wide=wide_dt.name, part=part, mods=mods):
                res = self.gen.fma(adreg=self.v1, bdreg=self.v2, cdreg=self.v0,
                                   a_dt=narrow_dt, b_dt=narrow_dt, c_dt=wide_dt,
                                   modifiers=modifiers,
                                   widening_method=wm.SPLIT_CONTIGUOUS
                                   if (adt.FP16, adt.FP32) == (narrow_dt, wide_dt)
                                   else wm.SPLIT_INTERLEAVED,
                                   part=part)
                self.assertEqual(res, expected)

    def test_indexed_fma(self):
        """Test indexed FMA operations (IDX)"""
        # Note: lane indexing depends on dt size. FP32 has 4 lanes (0-3).
        res = self.gen.fma(adreg=self.v1, bdreg=self.v2, cdreg=self.v0,
                           a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32,
                           operand_modifiers={'bdreg':{opd_mod.ILANE}}, bdreg_lane=2)
        self.assertEqual(res, "fmla v0.4s,v1.4s,v2.s[2]\n")

    def test_constraint_boundaries(self):
        """Test that out-of-bound immediates and registers are caught by constraints"""

        # 1. IDX out of bounds (FP32 max index is 3)
        with self.subTest(error="lane bounds"):
            with self.assertRaisesRegex(ValueError,
                                        "value of bdreg_lane must be <= 3"):
                self.gen.fma(adreg=self.v1, bdreg=self.v2, cdreg=self.v0,
                             a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32,
                             operand_modifiers={'bdreg':{opd_mod.ILANE}},
                             bdreg_lane=4)

        # 2. PART out of bounds (FP16 -> FP32 max part is 1)
        with self.subTest(error="part bounds"):
            with self.assertRaisesRegex(ValueError, "value of part must be <= 1"):
                self.gen.fma(adreg=self.v1, bdreg=self.v2, cdreg=self.v0,
                             a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP32,
                             modifiers={mod.PART},
                             widening_method=wm.SPLIT_CONTIGUOUS,
                             part=2)

        # 3. BDREG too high for 16-bit indexing (max v15)
        with self.subTest(error="bdreg 16-bit index bounds"):
            with self.assertRaisesRegex(ValueError, "index of bdreg must be <= 15"):
                self.gen.fma(adreg=self.v1, bdreg=self.v16, cdreg=self.v0,  # v16 is invalid here!
                             a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16,
                             operand_modifiers={'bdreg':{opd_mod.ILANE}},
                             bdreg_lane=1)

    def test_invalid_configurations(self):
        """Test mismatched types, missing kwargs, and unsupported forms"""
        # 1. Wrong register type (Scalar instead of Vector)
        with self.subTest(error="scalar register"):
            with self.assertRaisesRegex(ValueError, "Invalid configuration for neon_fma"):
                self.gen.fma(adreg=self.v1, bdreg=self.v2, cdreg=self.f0,
                             a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64)

        # 2. Missing 'part' kwarg when mod.PART is active
        with self.subTest(error="missing part kwarg"):
            with self.assertRaisesRegex(ValueError, "Operand missing: part"):
                self.gen.fma(adreg=self.v1, bdreg=self.v2, cdreg=self.v0,
                             a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP32,
                             widening_method=wm.SPLIT_CONTIGUOUS,
                             modifiers={mod.PART}) # part=... is missing

        # 3. MASK is unsupported in NEON
        with self.subTest(error="unsupported mask"):
            with self.assertRaisesRegex(ValueError, "NEON mul has no masked form"):
                self.gen.fma(adreg=self.v1, bdreg=self.v2, cdreg=self.v0,
                             a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32,
                             modifiers={mod.MASK})
