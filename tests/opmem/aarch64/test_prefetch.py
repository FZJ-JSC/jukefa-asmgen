# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Tests AArch64 prefetches
"""
import unittest

from asmgen.asmblocks.aarch64.opmem import aarch64_prefetch
from asmgen.asmblocks.aarch64.types import aarch64_greg
from asmgen.asmblocks.op import opmem_modifier as mod


def asmwrap(s: str) -> str:
    """
    Dummy asmwrap to reduce complexity
    """
    lines = s.split("\n")
    return "".join(f"{line}\n" for line in lines)

class test_aarch64_prefetch(unittest.TestCase):
    """
    Testsuite for AArch64 prefetches
    """
    def setUp(self):
        self.x0 = aarch64_greg(0)
        self.x1 = aarch64_greg(1)

        self.prefetch = aarch64_prefetch(asmwrap=asmwrap)

    def test_prefetch_without_dregs(self):
        """ Prefetch is a data-less opmem: no dregs, no dt """
        self.assertEqual(
            self.prefetch(areg=self.x0, modifiers=set()),
            "prfm PLDL1KEEP, [x0]\n"
        )

    def test_addressing_modes(self):
        """ Test offset addressing reuse """
        with self.subTest(mode="IOFFSET"):
            self.assertEqual(
                self.prefetch(areg=self.x0, modifiers={mod.IOFFSET},
                              ioffset=64),
                "prfm PLDL1KEEP, [x0, #64]\n"
            )
        with self.subTest(mode="GOFFSET"):
            self.assertEqual(
                self.prefetch(areg=self.x0, modifiers={mod.GOFFSET},
                              offreg=self.x1),
                "prfm PLDL1KEEP, [x0, x1]\n"
            )

    def test_invalid_configurations(self):
        """ Failure diagnosis is shared with the scalar opmem base """
        # GOFFSET without offreg; this one also proves the diagnosis is not
        # confused by data-less ops
        with self.assertRaisesRegex(ValueError, "Missing operand: offreg"):
            self.prefetch(areg=self.x0, modifiers={mod.GOFFSET})

        # unsupported modifier
        with self.assertRaisesRegex(ValueError,
                                    "Base AArch64 has no masked ld/st"):
            self.prefetch(areg=self.x0, modifiers={mod.MASK})