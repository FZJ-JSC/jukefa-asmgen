# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Contains the base class for AVX opd3 tests
"""
import unittest

from asmgen.asmblocks.avx import fma128,fma256,avx512

class test_avx_opd3(unittest.TestCase):
    """
    Tests AVX opd3 operations
    """

    def setUp(self):
        """
        Sets up the generators for all tests
        """
        self.gen128 = fma128()
        self.gen256 = fma256()
        self.gen512 = avx512()
        self.gen128.set_output_inline(yesno=False)
        self.gen256.set_output_inline(yesno=False)
        self.gen512.set_output_inline(yesno=False)
