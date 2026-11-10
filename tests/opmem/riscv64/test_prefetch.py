# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RISCV64 +D/F prefetch (Zicbop) testsuite
"""
import unittest

from asmgen.asmblocks.op import opmem_modifier as mod
from asmgen.asmblocks.riscv64.opmem import riscv64_prefetch
from asmgen.asmblocks.riscv64.types import riscv64_greg


def asmwrap(s: str) -> str:
    """
    Dummy asmwrap function to avoid pulling in more dependencies
    """
    lines = s.split("\n")
    return "".join(f"{line}\n" for line in lines)

class test_riscv64_prefetch(unittest.TestCase):
    """
    RISCV64 +D/F prefetch tests
    """
    def setUp(self):
        self.t1 = riscv64_greg(1)
        self.t2 = riscv64_greg(2)

        self.prefetch = riscv64_prefetch(asmwrap=asmwrap)

    def test_prefetch_without_dregs(self):
        """
        prefetch.r is a data-less opmem: no dregs, no dt
        """
        self.assertEqual(
            self.prefetch(areg=self.t1, modifiers=set()),
            "prefetch.r 0(t1)\n"
        )

    def test_addressing_modes(self):
        """
        Test offset addressing reuse
        """
        with self.subTest(mode="IOFFSET"):
            self.assertEqual(
                self.prefetch(areg=self.t1, modifiers={mod.IOFFSET},
                              ioffset=64),
                "prefetch.r 64(t1)\n"
            )

    def test_invalid_configurations(self):
        """
        Failure diagnosis is shared with the scalar opmem base
        """
        with self.assertRaisesRegex(ValueError,
                                    r"RISCV64 \+D/F has no masked ld/st"):
            self.prefetch(areg=self.t1, modifiers={mod.MASK})

        with self.assertRaisesRegex(
                ValueError,
                r"RISCV64 \+D/F has no ld/st with GP-reg strides"):
            self.prefetch(areg=self.t1, modifiers={mod.GSTRIDE},
                          streg=self.t2)
