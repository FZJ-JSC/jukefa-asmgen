# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Tests SVE loads/stores
"""
import unittest

from asmgen.asmblocks.op import (
    opmem_modifier as mod,
    operand_modifier as opd_mod
)
from asmgen.registers import asm_data_type as adt, asm_index_type as ait
from asmgen.asmblocks.aarch64.types import aarch64_greg, aarch64_freg
from asmgen.asmblocks.sve.types import sve_vreg, sve_preg
from asmgen.asmblocks.sve.opmem import sve_load, sve_store

def asmwrap(s: str) -> str:
    """
    Dummy asmwrap for testing
    """
    return f"{s}\n"

# It's fine
# pylint: disable-next=too-many-instance-attributes
class test_sve_opmem(unittest.TestCase):
    """
    Testsuite for SVE opmem instructions
    """
    def setUp(self):
        self.x0 = aarch64_greg(0)
        self.x1 = aarch64_greg(1)

        self.z0 = sve_vreg(0)
        self.z1 = sve_vreg(1)
        self.p1 = sve_preg(1)

        self.f0 = aarch64_freg(0, dt=adt.FP32)

        self.load = sve_load(asmwrap=asmwrap)
        self.store = sve_store(asmwrap=asmwrap)

    def test_scalar_routing(self):
        """Ensure scalar registers route back to aarch64_opmem"""
        self.assertEqual(
            self.load(dregs=[self.x1], areg=self.x0, dt=adt.UINT64, modifiers=set()),
            "ldr x1, [x0]\n"
        )
        self.assertEqual(
            self.load(dregs=[self.f0], areg=self.x0, dt=adt.FP32, modifiers=set()),
            "ldr s0, [x0]\n"
        )

    def test_basic_vector_ops(self):
        """Test standard unit-stride loads and stores (ld1w, st1d)"""
        self.assertEqual(
            self.load(dregs=[self.z0], areg=self.x0, amreg=self.p1,
                      dt=adt.FP32, modifiers={mod.MASK}),
            "ld1w {z0.s}, p1/z, [x0]\n"
        )
        self.assertEqual(
            self.store(dregs=[self.z1], areg=self.x0, amreg=self.p1,
                       dt=adt.FP64, modifiers={mod.MASK}),
            "st1d {z1.d}, p1, [x0]\n"
        )

    def test_addressing_modes(self):
        """Test standard offset scaling (VOFFSET, GOFFSET)"""
        with self.subTest(mode="VOFFSET"):
            res = self.load(dregs=[self.z0], areg=self.x0, amreg=self.p1, dt=adt.FP32,
                            modifiers={mod.MASK, mod.VOFFSET}, voffset=3)
            self.assertEqual(res, "ld1w {z0.s}, p1/z, [x0, #3, MUL VL]\n")

        with self.subTest(mode="GOFFSET"):
            res = self.load(dregs=[self.z0], areg=self.x0, amreg=self.p1, dt=adt.FP64,
                            modifiers={mod.MASK, mod.GOFFSET}, offreg=self.x1)
            self.assertEqual(res, "ld1d {z0.d}, p1/z, [x0, x1, lsl #3]\n")

    def test_vindex_gather(self):
        """Test VINDEX gather with element scaling matching data size"""
        with self.subTest(dt="FP32"):
            res = self.load(dregs=[self.z0], areg=self.x0, amreg=self.p1, dt=adt.FP32,
                            modifiers={mod.MASK, mod.VINDEX}, vidxreg=self.z1, it=ait.INT32)
            self.assertEqual(res, "ld1w {z0.s}, p1/z, [x0, z1.s, sxtw]\n")

        with self.subTest(dt="FP64"):
            res = self.load(dregs=[self.z0], areg=self.x0, amreg=self.p1, dt=adt.FP64,
                            modifiers={mod.MASK, mod.VINDEX}, vidxreg=self.z1, it=ait.INT64)
            self.assertEqual(res, "ld1d {z0.d}, p1/z, [x0, z1.d]\n")

    def test_broadcast_and_struct(self):
        """Test BCAST and STRUCT generation"""
        with self.subTest(mode="BCAST"):
            res = self.load(dregs=[self.z0], areg=self.x0, amreg=self.p1, dt=adt.FP32,
                            modifiers={mod.MASK},
                            operand_modifiers={'adreg' : {opd_mod.BCAST}})
            self.assertEqual(res, "ld1rw {z0.s}, p1/z, [x0]\n")

        with self.subTest(mode="STRUCT"):
            res = self.load(dregs=[self.z0, self.z1], areg=self.x0, amreg=self.p1, dt=adt.FP32,
                            modifiers={mod.MASK, mod.STRUCT}, nstructs=2)
            self.assertEqual(res, "ld2w {z0.s, z1.s}, p1/z, [x0]\n")

    def test_invalid_configurations(self):
        """Test that signatures correctly reject missing or mismatched arguments"""
        # 1. Missing amreg (Mask is required by signature for SVE ops)
        with self.subTest(error="missing amreg"):
            with self.assertRaisesRegex(ValueError, "Invalid configuration for sve_load"):
                self.load(dregs=[self.z0], areg=self.x0, dt=adt.FP32, modifiers={mod.MASK})

        # 2. Missing vidxreg
        with self.subTest(error="missing vidxreg"):
            with self.assertRaisesRegex(ValueError, "VINDEX modifier requires 'vidxreg' parameter"):
                self.load(dregs=[self.z0], areg=self.x0, amreg=self.p1, dt=adt.FP32,
                          modifiers={mod.MASK, mod.VINDEX}, it=ait.INT32)

        # 3. Mismatched Index Size
        with self.subTest(error="mismatched index size"):
            with self.assertRaisesRegex(ValueError, "Invalid configuration for sve_load"):
                self.load(dregs=[self.z0], areg=self.x0, amreg=self.p1, dt=adt.FP32,
                          modifiers={mod.MASK, mod.VINDEX}, vidxreg=self.z1, it=ait.INT64)

    def test_diagnose_failure_hooks(self):
        """Test the explicit diagnostic errors in the base class"""
        # 1. Mutually Exclusive Modifiers (Not generated in signatures, so it falls through)
        with self.subTest(error="VINDEX + VOFFSET"):
            with self.assertRaisesRegex(ValueError,
                                        "VINDEX cannot be combined with IOFFSET/VOFFSET"):
                self.load(dregs=[self.z0], areg=self.x0, amreg=self.p1, dt=adt.FP32,
                          modifiers={mod.MASK, mod.VINDEX, mod.VOFFSET},
                          vidxreg=self.z1, it=ait.INT32, voffset=4)

        # 2. BCAST on Store
        with self.subTest(error="BCAST on store"):
            with self.assertRaisesRegex(ValueError, "BCAST modifier is only valid for LOAD"):
                self.store(dregs=[self.z0], areg=self.x0, amreg=self.p1, dt=adt.FP32,
                           modifiers={mod.MASK},
                           operand_modifiers={'adreg' : {opd_mod.BCAST}})

        # 3. Unsupported ILANE
        with self.subTest(error="Unsupported ILANE"):
            with self.assertRaisesRegex(ValueError, "SVE has no immediate lane ld/st"):
                self.load(dregs=[self.z0], areg=self.x0, amreg=self.p1, dt=adt.FP32,
                          modifiers={mod.MASK},
                          operand_modifiers={'adreg':{opd_mod.ILANE}},
                          adreg_lane=1)


class test_struct_addressing(unittest.TestCase):
    """
    ld2d-ld4d, st2d-st4d at [x, #imm, MUL VL] - whole structures, -8..7 of
    them - and [x, x, lsl #e]: a loop over a panel's structures
    """
    def setUp(self):
        from asmgen.asmblocks.sve import sve
        self.g = sve()
        self.g.set_output_inline(False)

    def call(self, action, n, **extra):
        g = self.g
        kw = dict(dregs=[g.vreg(i) for i in range(n)], areg=g.greg(0), dt=adt.DOUBLE,
                  amreg=g.mreg(0), nstructs=n, **extra)
        return getattr(g, action), kw

    def test_voffset(self):
        op, kw = self.call("store", 4, modifiers={mod.MASK, mod.STRUCT, mod.VOFFSET}, voffset=8)
        self.assertEqual("st4d {z0.d, z1.d, z2.d, z3.d}, p0, [x0, #8, MUL VL]", op(**kw).strip())
        for bad in (6, 32, -36):                 # not a multiple of 4; past -32..28
            op, kw = self.call("store", 4, modifiers={mod.MASK, mod.STRUCT, mod.VOFFSET}, voffset=bad)
            self.assertIsNone(op.fits(**kw), bad)
        op, kw = self.call("load", 2, modifiers={mod.MASK, mod.STRUCT, mod.VOFFSET}, voffset=-16)
        self.assertEqual("ld2d {z0.d, z1.d}, p0/z, [x0, #-16, MUL VL]", op(**kw).strip())

    def test_goffset(self):
        op, kw = self.call("store", 3, modifiers={mod.MASK, mod.STRUCT, mod.GOFFSET}, offreg=self.g.greg(1))
        self.assertEqual("st3d {z0.d, z1.d, z2.d}, p0, [x0, x1, lsl #3]", op(**kw).strip())


if __name__ == '__main__':
    unittest.main()
