# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Test Base X86_64 prefetches
"""
import unittest

from asmgen.asmblocks.avx.types import reg_prefixer, x86_greg
from asmgen.asmblocks.op import opmem_modifier as mod
from asmgen.asmblocks.x86.opmem import x86_prefetch


class test_x86_prefetch(unittest.TestCase):
    """
    Testsuite for X86_64 prefetch instructions
    """
    def setUp(self):
        self.r8 = x86_greg(0)
        self.r15 = x86_greg(7)

        self.prefetch = x86_prefetch(asmwrap = lambda s : f"{s}\n",
                                     rpref=reg_prefixer(lambda: False))

    def test_prefetch_without_dregs(self):
        """ Prefetch is a data-less opmem: no dregs, no dt """
        self.assertEqual(
            self.prefetch(areg=self.r15, modifiers=set()),
            "prefetcht0 (%r15)\n"
        )

    def test_addressing_modes(self):
        """ Test offset addressing reuse """
        with self.subTest(mode="IOFFSET"):
            self.assertEqual(
                self.prefetch(areg=self.r15, modifiers={mod.IOFFSET},
                              ioffset=64),
                "prefetcht0 64(%r15)\n"
            )
        with self.subTest(mode="GOFFSET"):
            self.assertEqual(
                self.prefetch(areg=self.r15, modifiers={mod.GOFFSET},
                              offreg=self.r8),
                "prefetcht0 (%r15,%r8)\n"
            )

    def test_invalid_configurations(self):
        """ Test that invalid calls are rejected with useful errors """
        # dregs still require dt
        with self.assertRaisesRegex(ValueError,
                                    "dt is required when dregs are provided"):
            self.prefetch(areg=self.r15, dregs=[self.r8], modifiers=set())

        # GOFFSET without offreg
        with self.assertRaisesRegex(
                ValueError, "GOFFSET modifier requires 'offreg' parameter"):
            self.prefetch(areg=self.r15, modifiers={mod.GOFFSET})

        # unsupported modifier, diagnosed by the shared scalar base
        with self.assertRaisesRegex(ValueError,
                                    "Base X86_64 has no masked ld/st"):
            self.prefetch(areg=self.r15, modifiers={mod.MASK})