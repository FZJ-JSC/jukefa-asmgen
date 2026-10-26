# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Tests RVV fma instruction code generation
"""
import unittest

from asmgen.asmblocks.rvv import rvv
from asmgen.registers import asm_data_type as adt

from asmgen.asmblocks.op import optransform_modifier as mod
from asmgen.asmblocks.op import operand_modifier as opd_mod

from asmgen.asmblocks.op.optransform import widening_method as wm

class test_rvv_opd3(unittest.TestCase):
    """
    Tests RVV opd3 operations
    """
    def setUp(self):
        self.gen = rvv()
        self.gen.set_output_inline(yesno=False)

    def test_fma_fp64(self):
        """
        fp64 fma instruction
        """

        self.assertEqual(
                "vfmacc.vv v0,v2,v1\n",
                self.gen.fma(adreg=self.gen.vreg(1),
                             bdreg=self.gen.vreg(2),
                             cdreg=self.gen.vreg(0),
                             a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64))

    def test_mul_fp64(self):
        """
        fp64 fmul instruction
        """

        self.assertEqual(
                # Note: operand order different from FMA!
                "vfmul.vv v0,v1,v2\n",
                self.gen.mul(adreg=self.gen.vreg(1),
                              bdreg=self.gen.vreg(2),
                              cdreg=self.gen.vreg(0),
                              a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64))

    def test_fma_vf_fp64(self):
        """
        fp64 fma VF-form instruction
        """

        self.assertEqual(
                "vfmacc.vf v0,f2,v1\n",
                self.gen.fma(adreg=self.gen.vreg(1),
                             bdreg=self.gen.freg(2, dt=adt.FP64),
                             cdreg=self.gen.vreg(0),
                             a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64,
                             operand_modifiers={'bdreg':{opd_mod.VF}}))

    def test_mul_vf_fp64(self):
        """
        fp64 fmul VF-form instruction
        """

        self.assertEqual(
                "vfmul.vf v0,v1,f2\n",
                self.gen.mul(adreg=self.gen.vreg(1),
                             bdreg=self.gen.freg(2, dt=adt.FP64),
                             cdreg=self.gen.vreg(0),
                             a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64,
                             operand_modifiers={'bdreg':{opd_mod.VF}}))


    def test_mul_vf_fp64_a_is_freg(self):
        """
        fp64 fmul VF-form instruction with wrong dreg as freg
        """


        err_msg = "Invalid configuration for rvv_mul"
        with self.assertRaisesRegex(
                ValueError, err_msg):
            self.gen.mul(adreg=self.gen.freg(1, dt=adt.FP64),
                          bdreg=self.gen.vreg(2),
                          cdreg=self.gen.vreg(0),
                          a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64,
                          operand_modifiers={'bdreg':{opd_mod.VF}})

    def test_fma_fp32(self):
        """
        fp32 fma instruction
        """

        self.assertEqual(
                "vfmacc.vv v0,v2,v1\n",
                self.gen.fma(adreg=self.gen.vreg(1),
                             bdreg=self.gen.vreg(2),
                             cdreg=self.gen.vreg(0),
                             a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32))

    def test_fma_fp16(self):
        """
        fp16 fma instruction
        """

        self.assertEqual(
                "vfmacc.vv v0,v2,v1\n",
                self.gen.fma(adreg=self.gen.vreg(1),
                             bdreg=self.gen.vreg(2),
                             cdreg=self.gen.vreg(0),
                             a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16))


    def test_fma_fp16fp32(self):
        """
        fp16->fp32 widening instruction
        """
        self.assertEqual(
                "vfwmacc.vv v0,v3,v2\n",
                self.gen.fma(adreg=self.gen.vreg(2),
                             bdreg=self.gen.vreg(3),
                             cdreg=self.gen.vreg(0),
                             a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP32,
                             widening_method=wm.VEC_GROUP))


    def test_fma_4ways_widening(self):
        """
        fp8->fp32 widening instruction is invalid
        """
        # No more specific error message for 4-ways widening being invalid
        with self.assertRaisesRegex(
                ValueError, "Invalid configuration for rvv_fma"):
            self.gen.fma(adreg=self.gen.vreg(2),
                         bdreg=self.gen.vreg(3),
                         cdreg=self.gen.vreg(0),
                         a_dt=adt.FP8E5M2, b_dt=adt.FP8E5M2, c_dt=adt.FP32,
                         widening_method=wm.VEC_GROUP)

    def test_mul_np(self):
        """
        fp64 fmul does not support NP mod
        """
        with self.assertRaisesRegex(
                ValueError, "rvv_mul does not support these modifiers at all: {NP}"):
            self.gen.mul(adreg=self.gen.vreg(2),
                          bdreg=self.gen.vreg(3),
                          cdreg=self.gen.vreg(0),
                          a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64,
                          modifiers={mod.NP})


    def test_fma_sint32(self):
        """
        int32 fma instruction
        """

        self.assertEqual(
                "vmacc.vv v0,v2,v1\n",
                self.gen.fma(adreg=self.gen.vreg(1),
                             bdreg=self.gen.vreg(2),
                             cdreg=self.gen.vreg(0),
                             a_dt=adt.SINT32, b_dt=adt.SINT32, c_dt=adt.SINT32))

    def test_fma_vf_fp16fp32(self):
        """
        fp16->fp32 widening negated-product VF-form instruction
        """
        self.assertEqual(
                "vfwnmsac.vf v0,f0,v2\n",
                self.gen.fma(adreg=self.gen.vreg(2),
                             bdreg=self.gen.freg(0, adt.FP16),
                             cdreg=self.gen.vreg(0),
                             a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP32,
                             modifiers={mod.NP},
                             operand_modifiers={'bdreg':{opd_mod.VF}},
                             widening_method=wm.VEC_GROUP))

    def test_fma_vf_int32_np(self):
        """
        int16->int32 widening negated-product VF-form instruction is invalid in RVV
        """
        self.assertEqual(
                "vnmsac.vx v0,t0,v2\n",
                self.gen.fma(adreg=self.gen.vreg(2),
                             bdreg=self.gen.greg(0),
                             cdreg=self.gen.vreg(0),
                             a_dt=adt.SINT32, b_dt=adt.SINT32, c_dt=adt.SINT32,
                             modifiers={mod.NP},
                             operand_modifiers={'bdreg':{opd_mod.VF}}))

    def test_fma_vf_int16int32_np_invalid(self):
        """
        int16->int32 widening negated-product VF-form instruction is invalid in RVV
        """
        with self.assertRaisesRegex(ValueError, "Invalid configuration for rvv_fma"):
            self.gen.fma(adreg=self.gen.vreg(2),
                         bdreg=self.gen.greg(0),
                         cdreg=self.gen.vreg(0),
                         a_dt=adt.SINT16, b_dt=adt.SINT16, c_dt=adt.SINT32,
                         modifiers={mod.NP},
                         operand_modifiers={'bdreg':{opd_mod.VF}},
                         widening_method=wm.VEC_GROUP)


    def test_wrong_registers(self):
        """
        Tests that the correct error is raised if wrong registers are passed
        to the operation
        """
        err_msg = "Invalid configuration for rvv_fma"
        with self.assertRaisesRegex(
                ValueError, err_msg):
            self.gen.fma(
                adreg=self.gen.vreg(1),
                bdreg=self.gen.vreg(2),
                cdreg=self.gen.freg(0,dt=adt.FP64),
                a_dt=adt.FP64,b_dt=adt.FP64,c_dt=adt.FP64)

        with self.assertRaisesRegex(
                ValueError, err_msg):
            self.gen.fma(
                adreg=self.gen.greg(1),
                bdreg=self.gen.vreg(2),
                cdreg=self.gen.freg(0,dt=adt.FP64),
                a_dt=adt.FP64,b_dt=adt.FP64,c_dt=adt.FP64)

        with self.assertRaisesRegex(
                ValueError, err_msg):
            self.gen.fma(
                adreg=self.gen.freg(1,dt=adt.FP64),
                bdreg=self.gen.vreg(2),
                cdreg=self.gen.vreg(0),
                a_dt=adt.FP64,b_dt=adt.FP64,c_dt=adt.FP64)
