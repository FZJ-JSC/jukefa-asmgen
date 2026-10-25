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
