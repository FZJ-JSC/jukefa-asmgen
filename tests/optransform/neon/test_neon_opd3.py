# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Contains the base class for NEON opd3 tests
"""
import unittest

from asmgen.asmblocks.neon import neon
from asmgen.registers import asm_data_type as adt

class test_neon_opd3_base(unittest.TestCase):
    """Base setup for NEON opd3 tests"""
    def setUp(self):
        self.gen = neon()
        self.gen.set_output_inline(False)

        # Standard vector registers
        self.v0 = self.gen.vreg(0)
        self.v1 = self.gen.vreg(1)
        self.v2 = self.gen.vreg(2)
        self.v16 = self.gen.vreg(16) # Used for testing indexed register limits

        # Scalar register
        self.f0 = self.gen.freg(0, dt=adt.FP32)
