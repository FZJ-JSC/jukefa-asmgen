# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Contains the base class for SVE opd3 tests
"""
import unittest

from asmgen.asmblocks.sve import sve

# It's fine for the testing
# pylint: disable-next=too-many-instance-attributes
class test_sve_opd3_base(unittest.TestCase):
    """
    Base setup for SVE opd3 tests
    """
    def setUp(self):
        self.gen = sve()
        self.gen.set_output_inline(False)

        # Standard vector registers
        self.v0 = self.gen.vreg(0)
        self.v1 = self.gen.vreg(1)
        self.v2 = self.gen.vreg(2)

        # Upper registers (crucial for testing SVE indexing constraints)
        self.v7 = self.gen.vreg(7)
        self.v8 = self.gen.vreg(8)
        self.v15 = self.gen.vreg(15)
        self.v16 = self.gen.vreg(16)

        # Predicate register
        self.p0 = self.gen.mreg(0)
