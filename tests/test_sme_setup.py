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
    Registers clobbered beyond the allocation, kept by the calling
    convention where they are the callee's
    """

    def test_sme_setup_clobbers_vectors(self):
        from asmgen.asmblocks.setup import sme_setup
        g = sme()
        self.assertEqual(list(range(g.max_fregs())), sme_setup(g).clobbers()["freg"])

    def test_callconv_keeps_callee_saved_clobbers(self):
        g = sme()
        g.output_inline = False
        cc = g.create_callconv()
        cc.add_clobber("freg", range(32))
        save = cc.save_in_call(g, regs={"greg": [], "freg": []})
        # AAPCS64: d8-d15 the callee's, the others not
        self.assertEqual(sorted(f"d{n}" for n in range(8, 16)),
                         sorted(l.split()[1].split(",")[0] for l in save.splitlines()
                                if l.strip().startswith("str")))
