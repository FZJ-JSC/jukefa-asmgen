"""
The register tracker's available(tag) (0.6, phase 7): the registers
reserve_any_reg could still give - shared files counted - instead of
reserving until it raises; available_reg_count stays the allocatable total.
Every generator's max_*regs are methods returning counts.
"""
import unittest

from asmgen.asmblocks.avx import fma128, fma256, avx512
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.sme import sme
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.rvv071 import rvv071
from asmgen.registers import asm_data_type as adt

GENERATORS = (fma128, fma256, avx512, neon, sve, sme, rvv, rvv071)


class test_available(unittest.TestCase):
    def test_counts_down(self):
        rt = neon().get_reg_tracker()
        free, allocatable = rt.available("greg"), rt.available_reg_count("greg")
        self.assertLessEqual(free, allocatable)          # some are reserved from the start
        rt.reserve_any_reg("greg")
        self.assertEqual(free - 1, rt.available("greg"))
        self.assertEqual(allocatable, rt.available_reg_count("greg"))

    def test_shared_files(self):
        rt = fma256().get_reg_tracker()               # AVX2's masks are vector registers
        vregs, mregs = rt.available("vreg"), rt.available("mreg")
        rt.reserve_any_reg("vreg")
        self.assertEqual((vregs - 1, mregs - 1), (rt.available("vreg"), rt.available("mreg")))
        rt = sve().get_reg_tracker()                  # FP registers are vector registers
        vregs = rt.available("vreg")
        rt.reserve_any_reg("freg")
        self.assertEqual(vregs - 1, rt.available("vreg"))

    def test_to_exhaustion(self):
        rt = rvv().get_reg_tracker()
        for _ in range(rt.available("vreg")):
            rt.reserve_any_reg("vreg")
        self.assertEqual(0, rt.available("vreg"))
        with self.assertRaises(IndexError):
            rt.reserve_any_reg("vreg")

    def test_unknown_type(self):
        with self.assertRaises(ValueError):
            neon().get_reg_tracker().available("nosuchtype")


class test_pools(unittest.TestCase):
    """
    pool(tag): the types whose registers are the same registers (share);
    capacity(tags): how many registers they are together
    """
    def test_files(self):
        rt = sve().get_reg_tracker()                  # FP registers are vector registers
        self.assertEqual({"freg", "vreg"}, rt.pool("freg"))
        self.assertEqual(32, rt.capacity(rt.pool("vreg")))
        rt = rvv().get_reg_tracker()                  # own FP registers; v0 is the mask
        self.assertEqual({"freg"}, rt.pool("freg"))
        self.assertEqual({"mreg", "vreg"}, rt.pool("vreg"))
        self.assertEqual(32, rt.capacity(rt.pool("vreg")))
        rt = fma256().get_reg_tracker()               # AVX2: FP, masks and vectors one file
        self.assertEqual({"freg", "mreg", "vreg"}, rt.pool("mreg"))
        self.assertEqual(16, rt.capacity(rt.pool("vreg")))
        rt = avx512().get_reg_tracker()               # k registers apart
        self.assertEqual({"mreg"}, rt.pool("mreg"))

    def test_capacity_counts_shared_once(self):
        rt = rvv().get_reg_tracker()
        self.assertEqual(32 + 32, rt.capacity({"vreg", "freg"}))   # apart
        rt = neon().get_reg_tracker()
        self.assertEqual(32, rt.capacity({"vreg", "freg"}))        # one file

    def test_clobbers_through_shared(self):
        rt = neon().get_reg_tracker()
        rt.reserve_specific_reg("vreg", 9)
        rt.clobbered_regs["vreg"].add(10)             # recorded however: still the d register's
        self.assertTrue({9, 10} <= rt.get_clobbered_regs("freg", through_shared=True))
        rt = rvv().get_reg_tracker()
        rt.reserve_specific_reg("vreg", 9)
        self.assertNotIn(9, rt.get_clobbered_regs("freg", through_shared=True))


class test_counts_are_methods(unittest.TestCase):
    def test_every_generator(self):
        for gen in GENERATORS:
            g = gen()
            with self.subTest(gen=gen.__name__):
                for name in ("max_gregs", "max_vregs", "max_fregs", "max_mregs"):
                    self.assertIsInstance(getattr(g, name)(), int, name)
                self.assertIsInstance(g.max_tregs(adt.FP64), int)


if __name__ == "__main__":
    unittest.main()
