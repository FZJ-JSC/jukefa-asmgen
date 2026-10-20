"""
SME: tiles zeroed as lists; the SME2 predicate-as-counter set only with SME2
"""

import unittest

from asmgen.asmblocks.sme import sme
from asmgen.registers import asm_data_type as adt


class test_sme_setup(unittest.TestCase):

    def test_zero_list(self):
        g = sme()
        g.output_inline = False
        self.assertEqual("zero {za1.d}", g.zero_treg(treg=g.treg(1, adt.FP64), dt=adt.FP64).strip())

    def test_counter_with_sme2(self):
        from asmgen.asmblocks.setup import sme_setup
        g = sme()
        g.output_inline = False
        g.set_features([])
        self.assertFalse(sme_setup(g).counter())
        g.set_features(["SME2"])
        self.assertTrue(sme_setup(g).counter())


if __name__ == "__main__":
    unittest.main()


class test_clobbers(unittest.TestCase):
    """
    Registers clobbered beyond the allocation: entering streaming mode
    clobbers every vector register
    """

    def test_sme_setup_clobbers_vectors(self):
        from asmgen.asmblocks.setup import sme_setup
        g = sme()
        self.assertEqual(list(range(g.max_fregs())), sme_setup(g).clobbers()["freg"])


if __name__ == "__main__":
    unittest.main()
