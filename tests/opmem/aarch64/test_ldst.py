# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Tests AArch64 loads/stores
"""
import unittest

from asmgen.asmblocks.aarch64.opmem import aarch64_load, aarch64_store
from asmgen.asmblocks.aarch64.types import aarch64_freg, aarch64_greg
from asmgen.asmblocks.neon.types import neon_vreg
from asmgen.asmblocks.op import operand_modifier as opd_mod
from asmgen.asmblocks.op import opmem_modifier as mod
from asmgen.registers import asm_data_type as adt


def asmwrap(s: str) -> str:
    """
    Dummy asmwrap to reduce complexity
    """
    lines = s.split("\n")
    return "".join(f"{line}\n" for line in lines)

# It's fine for this test
# pylint: disable-next=too-many-instance-attributes
class test_aarch64_opmem(unittest.TestCase):
    """
    Testsuite for AArch64 loads and stores
    """

    def setUp(self):
        # General purpose registers
        self.x0 = aarch64_greg(0)
        self.x1 = aarch64_greg(1)
        self.x2 = aarch64_greg(2)

        # Floating point registers (various sizes)
        self.f0_32 = aarch64_freg(0, adt.FP32)
        self.f1_64 = aarch64_freg(1, adt.FP64)
        self.f2_16 = aarch64_freg(2, adt.FP16)

        # Vector register (for testing failures)
        self.v0 = neon_vreg(0)

        self.load = aarch64_load(asmwrap=asmwrap)
        self.store = aarch64_store(asmwrap=asmwrap)

    # --- 1. Basic Load/Store and Size Mnemonics ---

    def test_basic_loads(self):
        """ Test scalar zero-offset loads across data types and sizes """
        # FP64 -> ldr
        self.assertEqual(
            self.load(dregs=[self.f1_64], areg=self.x0, dt=adt.FP64, modifiers=set()),
            "ldr d1, [x0]\n"
        )
        # FP32 -> ldr
        self.assertEqual(
            self.load(dregs=[self.f0_32], areg=self.x0, dt=adt.FP32, modifiers=set()),
            "ldr s0, [x0]\n"
        )
        # UINT8 (1 byte) -> ldrb
        self.assertEqual(
            self.load(dregs=[self.x1], areg=self.x0, dt=adt.UINT8, modifiers=set()),
            "ldrb w1, [x0]\n"
        )
        # UINT16 (2 bytes) -> ldrh
        self.assertEqual(
            self.load(dregs=[self.x1], areg=self.x0, dt=adt.UINT16, modifiers=set()),
            "ldrh w1, [x0]\n"
        )
        # UINT64 (8 bytes) -> ldr
        self.assertEqual(
            self.load(dregs=[self.x1], areg=self.x0, dt=adt.UINT64, modifiers=set()),
            "ldr x1, [x0]\n"
        )

    def test_basic_stores(self):
        """ Test scalar zero-offset stores across data types and sizes """
        self.assertEqual(
            self.store(dregs=[self.f1_64], areg=self.x0, dt=adt.FP64, modifiers=set()),
            "str d1, [x0]\n"
        )
        self.assertEqual(
            self.store(dregs=[self.x1], areg=self.x0, dt=adt.UINT8, modifiers=set()),
            "strb w1, [x0]\n"
        )
        self.assertEqual(
            self.store(dregs=[self.x1], areg=self.x0, dt=adt.UINT16, modifiers=set()),
            "strh w1, [x0]\n"
        )

    # --- 2. Addressing Modes & Modifiers ---

    def test_immediate_offset(self):
        """ Test loading/storing with an immediate offset """
        self.assertEqual(
            self.load(dregs=[self.f0_32], areg=self.x0, dt=adt.FP32,
                      modifiers={mod.IOFFSET}, ioffset=16),
            "ldr s0, [x0, #16]\n"
        )
        # Test zero offset optimization
        self.assertEqual(
            self.store(dregs=[self.x1], areg=self.x0, dt=adt.UINT64,
                       modifiers={mod.IOFFSET}, ioffset=0),
            "str x1, [x0]\n"
        )

    def test_register_offset_goffset(self):
        """ Test loading with a register offset [Xn, Xm] """
        self.assertEqual(
            self.load(dregs=[self.x1], areg=self.x0, dt=adt.UINT64,
                      modifiers={mod.GOFFSET}, offreg=self.x2),
            "ldr x1, [x0, x2]\n"
        )

    def test_post_increment(self):
        """ Test post-index addressing [Xn], #imm and [Xn], Xm """
        # Immediate post-inc
        self.assertEqual(
            self.load(dregs=[self.x1], areg=self.x0, dt=adt.UINT64,
                      modifiers={mod.POSTINC}, iinc=8),
            "ldr x1, [x0], #8\n"
        )
        # Register post-inc
        self.assertEqual(
            self.store(dregs=[self.x1], areg=self.x0, dt=adt.UINT64,
                       modifiers={mod.POSTINC}, increg=self.x2),
            "str x1, [x0], x2\n"
        )

    # --- 3. Error Handling and Input Validation ---

    def test_invalid_modifiers(self):
        """ Test that invalid modifiers raise ValueError """
        invalid_mods = [mod.TINDEX, mod.VINDEX,
                        mod.TOFFSET, mod.VOFFSET, mod.ISTRIDE, mod.GSTRIDE,
                        mod.STRUCT]

        invalid_opd_mods = [opd_mod.ILANE,opd_mod.GLANE,
                            opd_mod.VF,opd_mod.BCAST]

        for invalid_mod in invalid_mods:
            for invalid_opd_mod in invalid_opd_mods:
                with self.subTest(modifier=invalid_mod,
                                  opd_modifier=invalid_opd_mod):
                    with self.assertRaisesRegex(ValueError, "Base AArch64 has no"):
                        self.load(dregs=[self.x1], areg=self.x0,
                                  dt=adt.UINT64,
                                  modifiers={invalid_mod},
                                  operand_modifiers={'adreg':invalid_opd_mod})

    def test_missing_required_parameters(self):
        """
        Test that omitting required kwargs for modifiers raises ValueError """
        with self.assertRaisesRegex(ValueError,
                                    "Missing operand: ioffset"):
            self.load(dregs=[self.x1], areg=self.x0, dt=adt.UINT64, modifiers={mod.IOFFSET})

        with self.assertRaisesRegex(ValueError, "Missing operand: offreg"):
            self.load(dregs=[self.x1], areg=self.x0, dt=adt.UINT64, modifiers={mod.GOFFSET})

        with self.assertRaisesRegex(ValueError, "Missing one of these parameters: iinc, increg"):
            self.load(dregs=[self.x1], areg=self.x0, dt=adt.UINT64, modifiers={mod.POSTINC})

    def test_mutually_exclusive_parameters(self):
        """ Test that providing both post-inc parameters raises ValueError """
        with self.assertRaisesRegex(ValueError, "iinc, increg are mutually exclusive"):
            self.load(dregs=[self.x1], areg=self.x0, dt=adt.UINT64,
                      modifiers={mod.POSTINC}, iinc=8, increg=self.x2)

    def test_invalid_register_types(self):
        """ Test that passing a vreg (vector) or non-greg address raises errors """
        # Bad DREG
        with self.assertRaisesRegex(ValueError,
                                    "Invalid configuration for aarch64_load"):
            self.load(dregs=[self.v0], areg=self.x0, dt=adt.UINT64, modifiers=set())

        # Bad AREG
        with self.assertRaisesRegex(ValueError,
                                    "Invalid configuration for aarch64_load"):
            self.load(dregs=[self.x1], areg=self.f0_32, dt=adt.UINT64, modifiers=set())

    def test_invalid_register_count(self):
        """ Test that providing too many or too few dregs raises an error """
        with self.assertRaisesRegex(
                ValueError,
                "AArch64 scalar load/store uses exactly one register."):
            self.load(dregs=[self.x1, self.x2], areg=self.x0, dt=adt.UINT64, modifiers=set())

        with self.assertRaisesRegex(
                ValueError,
                "AArch64 scalar load/store uses exactly one register."):
            self.load(dregs=[], areg=self.x0, dt=adt.UINT64, modifiers=set())

if __name__ == '__main__':
    unittest.main()
