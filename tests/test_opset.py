"""
The set operation (docs/V06_REWORK.md, phase 4): CONST zero on general,
floating-point and vector registers and tiles; the zero_* methods on it
"""
import unittest

from asmgen.asmblocks.avx import avx512, fma256
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.op.opset import opset_modifier
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.sme import sme
from asmgen.asmblocks.sve import sve
from asmgen.registers import asm_data_type as adt

CONST = {opset_modifier.CONST}


def make(ctor):
    g = ctor()
    g.set_output_inline(yesno=False)
    return g


class test_setop(unittest.TestCase):
    def test_zero_methods_on_set(self):
        for ctor in (rvv, neon, sve, sme, fma256, avx512):
            g = make(ctor)
            with self.subTest(isa=ctor.__name__):
                self.assertEqual(g.set(dreg=g.greg(1), dt=adt.UINT64, modifiers=CONST, constant="zero"),
                                 g.zero_greg(greg=g.greg(1)))
                f = g.freg(1, adt.SINGLE)
                self.assertEqual(g.set(dreg=f, dt=adt.SINGLE, modifiers=CONST, constant="zero"),
                                 g.zero_freg(freg=f, dt=adt.SINGLE))
                self.assertEqual(g.set(dreg=g.vreg(2), dt=adt.DOUBLE, modifiers=CONST, constant="zero"),
                                 g.zero_vreg(vreg=g.vreg(2), dt=adt.DOUBLE))
        g = make(sme)
        t = g.treg(1, adt.DOUBLE)
        self.assertEqual("zero {za1.d}", g.set(dreg=t, dt=adt.DOUBLE, modifiers=CONST, constant="zero").strip())
        self.assertEqual(g.set(dreg=t, dt=adt.DOUBLE, modifiers=CONST, constant="zero"), g.zero_treg(treg=t, dt=adt.DOUBLE))

    def test_unknown_constant(self):
        g = make(rvv)
        with self.assertRaises(ValueError):
            g.set(dreg=g.vreg(1), dt=adt.DOUBLE, modifiers=CONST, constant="pi")

    def test_tiles_only_where_there_are_tiles(self):
        kinds = lambda g: {o.rtype.name for s in g.set.get_signatures() for o in s.operands.values()}
        self.assertIn("TILE", kinds(make(sme)))
        self.assertNotIn("TILE", kinds(make(sve)))
        self.assertTrue({"GP", "FP", "VEC"} <= kinds(make(rvv)))


if __name__ == "__main__":
    unittest.main()
