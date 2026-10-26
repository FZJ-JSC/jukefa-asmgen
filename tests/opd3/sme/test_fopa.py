# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Tests SME fopa instruction code generation
"""
import unittest

from asmgen.asmblocks.sme import sme
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.op import (
    optransform_modifier as mod,
    operand_modifier as opd_mod
)

from asmgen.asmblocks.op.optransform import widening_method as wm

class test_sme_opd3(unittest.TestCase):
    """
    Tests SME opd3 operations
    """

    def setUp(self):
        """
        Initialize the generator before each test to reduce boilerplate.
        """
        self.gen = sme()
        self.gen.set_output_inline(yesno=False)
        self.za0 = self.gen.treg(0, adt.FP64) # Default, will override dt in tests
        self.z0 = self.gen.vreg(0)
        self.z1 = self.gen.vreg(1)
        self.p0 = self.gen.mreg(0)
        self.p1 = self.gen.mreg(1)

    # ---------------------------------------------------------
    # Float Valid Operations
    # ---------------------------------------------------------

    def test_fopa_float_same_size(self):
        """Tests standard same-size float outer product accumulate (fmopa)"""
        # FP64
        self.assertEqual(
            "fmopa za0.d,p0/m,p1/m,z0.d,z1.d\n",
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.FP64),
                          amreg=self.p0, bmreg=self.p1,
                          a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64,
                          modifiers={mod.MASK})
        )
        # FP32
        self.assertEqual(
            "fmopa za0.s,p0/m,p1/m,z0.s,z1.s\n",
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.FP32),
                          amreg=self.p0, bmreg=self.p1,
                          a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32,
                          modifiers={mod.MASK})
        )
        # FP16
        self.assertEqual(
            "fmopa za0.h,p0/m,p1/m,z0.h,z1.h\n",
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.FP16),
                          amreg=self.p0, bmreg=self.p1,
                          a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16,
                          modifiers={mod.MASK})
        )

    def test_fopa_float_widening(self):
        """Tests widening float outer product accumulate"""
        # FP16 -> FP32
        self.assertEqual(
            "fmopa za0.s,p0/m,p0/m,z0.h,z1.h\n",
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.FP32),
                          amreg=self.p0, bmreg=self.p0,
                          a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP32,
                          modifiers={mod.MASK}, widening_method=wm.DOT_NEIGHBOURS)
        )

    # ---------------------------------------------------------
    # Integer Valid Operations
    # ---------------------------------------------------------

    def test_fopa_int_signed(self):
        """Tests signed integer outer product accumulate (smopa)"""
        # SINT16 -> SINT32
        self.assertEqual(
            "smopa za0.s,p0/m,p0/m,z0.h,z1.h\n",
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.SINT32),
                          amreg=self.p0, bmreg=self.p0,
                          a_dt=adt.SINT16, b_dt=adt.SINT16, c_dt=adt.SINT32,
                          modifiers={mod.MASK}, widening_method=wm.DOT_NEIGHBOURS)
        )
        # SINT8 -> SINT32
        self.assertEqual(
            "smopa za0.s,p0/m,p0/m,z0.b,z1.b\n",
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.SINT32),
                          amreg=self.p0, bmreg=self.p0,
                          a_dt=adt.SINT8, b_dt=adt.SINT8, c_dt=adt.SINT32,
                          modifiers={mod.MASK}, widening_method=wm.DOT_NEIGHBOURS)
        )

    def test_fopa_int_unsigned(self):
        """Tests unsigned integer outer product accumulate (umopa)"""
        # UINT16 -> UINT32
        self.assertEqual(
            "umopa za0.s,p0/m,p0/m,z0.h,z1.h\n",
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.UINT32),
                          amreg=self.p0, bmreg=self.p0,
                          a_dt=adt.UINT16, b_dt=adt.UINT16, c_dt=adt.UINT32,
                          modifiers={mod.MASK}, widening_method=wm.DOT_NEIGHBOURS)
        )

    def test_fopa_mixed_sign_int(self):
        """
        Tests the mixed sign INT constraints.
        """
        # SINT16,UINT16 -> SINT64
        self.assertEqual(
            "sumopa za0.d,p0/m,p0/m,z0.h,z1.h\n",
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.SINT64),
                          amreg=self.p0, bmreg=self.p0,
                          a_dt=adt.SINT16, b_dt=adt.UINT16, c_dt=adt.SINT64,
                          modifiers={mod.MASK}, widening_method=wm.DOT_NEIGHBOURS)
        )

    # ---------------------------------------------------------
    # Modifiers
    # ---------------------------------------------------------

    def test_fopa_subtract_modifier(self):
        """Tests that the Negative Product (NP) modifier generates subtract instructions (mops)"""
        # Float
        self.assertEqual(
            "fmops za0.d,p0/m,p0/m,z0.d,z1.d\n",
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.FP64),
                          amreg=self.p0, bmreg=self.p0,
                          a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64,
                          modifiers={mod.MASK, mod.NP})
        )
        # Signed Int
        self.assertEqual(
            "smops za0.s,p0/m,p0/m,z0.h,z1.h\n",
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.SINT32),
                          amreg=self.p0, bmreg=self.p0,
                          a_dt=adt.SINT16, b_dt=adt.SINT16, c_dt=adt.SINT32,
                          modifiers={mod.MASK, mod.NP},
                          widening_method=wm.DOT_NEIGHBOURS)
        )

    # ---------------------------------------------------------
    # Exceptions and Error Handling
    # ---------------------------------------------------------

    def test_fopa_invalid_modifier_ilane(self):
        """Tests that the ILANE modifier raises an error for SME fopa."""
        with self.assertRaisesRegex(ValueError, r"SME has no immediate lane opd3"):
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.za0,
                          a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64,
                          operand_modifiers={'adreg':{opd_mod.ILANE}},
                          adreg_lane=0)

    def test_fopa_invalid_modifier_part(self):
        """Tests that the PART modifier raises an error for SME fopa."""
        expected_error = "SME has no partial instructions"
        with self.assertRaisesRegex(ValueError, expected_error):
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.za0,
                          a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64,
                          modifiers={mod.PART})

    def test_fopa_invalid_modifier_vf(self):
        """Tests that the VF modifier raises an error for SME fopa."""
        with self.assertRaisesRegex(ValueError, r"SME has no VF opd3"):
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.za0,
                          a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64,
                          operand_modifiers={'adreg':{opd_mod.VF}})

    def test_fopa_invalid_registers(self):
        """Tests that passing the wrong register types (e.g. vreg for ZA) raises errors"""
        # C register is not an sme_treg
        with self.assertRaisesRegex(ValueError, "Invalid configuration for sme_fopa"):
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.vreg(0),
                          a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64)

        # A register is not an sve_vreg
        with self.assertRaisesRegex(ValueError, "Invalid configuration for sme_fopa"):
            self.gen.fopa(adreg=self.za0, bdreg=self.z1, cdreg=self.za0,
                          a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64)

    def test_fopa_invalid_types_and_sizes(self):
        """Tests logical restrictions on datatypes inside __call__"""

        # C is smaller than A/B
        with self.assertRaisesRegex(ValueError,
                "Invalid configuration for sme_fopa"):
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.FP16),
                          a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP16)

        # Mixing Float and Int
        with self.assertRaisesRegex(ValueError,
                "Invalid configuration for sme_fopa"):
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.UINT64),
                          a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.UINT64)

        # C is not in valid_c_types (e.g. UINT8)
        with self.assertRaisesRegex(ValueError,
                "Invalid configuration for sme_fopa"):
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.UINT8),
                          a_dt=adt.UINT8, b_dt=adt.UINT8, c_dt=adt.UINT8)
