# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Tests AVX fmla instruction code generation
"""
from asmgen.registers import asm_data_type as adt

from .test_avx_opd3 import test_avx_opd3

class test_avx_fma(test_avx_opd3):
    """
    Tests AVX opd3 operations
    """
    def test_fp64(self):
        """
        Tests that the AVX generators generate correct FP64 FMA instructions
        """

        self.assertEqual(
            "vfmadd231pd %xmm1,%xmm2,%xmm0\n",
            self.gen128.fma(
                adreg=self.gen128.vreg(1),
                bdreg=self.gen128.vreg(2),
                cdreg=self.gen128.vreg(0),
                a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64))

        self.assertEqual(
            "vfmadd231pd %ymm1,%ymm2,%ymm0\n",
            self.gen256.fma(
                adreg=self.gen256.vreg(1),
                bdreg=self.gen256.vreg(2),
                cdreg=self.gen256.vreg(0),
                a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64))

        self.assertEqual(
            "vfmadd231pd %zmm1,%zmm2,%zmm0\n",
            self.gen512.fma(
                adreg=self.gen512.vreg(1),
                bdreg=self.gen512.vreg(2),
                cdreg=self.gen512.vreg(0),
                a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64))

    def test_fp32(self):
        """
        Tests that the AVX generators generate correct FP32 FMA instructions
        """

        self.assertEqual(
            "vfmadd231ps %xmm1,%xmm2,%xmm0\n",
            self.gen128.fma(
                adreg=self.gen128.vreg(1),
                bdreg=self.gen128.vreg(2),
                cdreg=self.gen128.vreg(0),
                a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32))

        self.assertEqual(
            "vfmadd231ps %ymm1,%ymm2,%ymm0\n",
            self.gen256.fma(
                adreg=self.gen256.vreg(1),
                bdreg=self.gen256.vreg(2),
                cdreg=self.gen256.vreg(0),
                a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32))

        self.assertEqual(
            "vfmadd231ps %zmm1,%zmm2,%zmm0\n",
            self.gen512.fma(
                adreg=self.gen512.vreg(1),
                bdreg=self.gen512.vreg(2),
                cdreg=self.gen512.vreg(0),
                a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32))

    def test_fp16(self):
        """
        Tests that the AVX generators generate correct FP16 FMA instructions
        """
        self.assertEqual(
            "vfmadd231ph %zmm1,%zmm2,%zmm0\n",
            self.gen512.fma(
                adreg=self.gen512.vreg(1),
                bdreg=self.gen512.vreg(2),
                cdreg=self.gen512.vreg(0),
                a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16))

    def test_wrong_registers(self):
        """
        Tests that the correct error is raised if wrong registers are passed
        to the operation
        """
        with self.assertRaisesRegex(
                ValueError,
                "Invalid configuration for avx_fma"):
            self.gen128.fma(
                    adreg=self.gen128.vreg(1),
                    bdreg=self.gen128.vreg(2),
                    cdreg=self.gen128.freg(0,dt=adt.FP64),
                    a_dt=adt.FP64,b_dt=adt.FP64,c_dt=adt.FP64)
