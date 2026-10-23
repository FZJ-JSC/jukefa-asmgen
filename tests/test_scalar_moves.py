"""
Moves between scalar registers as operations (docs/V06_REWORK.md, phase 3):
move([greg, greg]), move([freg, freg]), move([greg, freg], BITS); the old
methods on top of them
"""
import unittest

from asmgen.asmblocks.avx import avx512
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.op.move import move_modifier
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.sme import sme
from asmgen.asmblocks.sve import sve
from asmgen.registers import asm_data_type as adt

U = adt.UINT64
BITS = {move_modifier.BITS}


def make(ctor):
    g = ctor()
    g.set_output_inline(yesno=False)
    return g


class test_scalar_moves(unittest.TestCase):
    def test_texts(self):
        expect = {
            rvv: ("add t2,t1,0", "fmv.d f2,f1", "fmv.d.x f2,t1", "fmv.w.x f2,t1"),
            neon: ("mov x2,x1", "fmov d2,d1", "fmov d2,x1", "fmov s2,w1"),
            sme: ("mov x2,x1", "fmov d2,d1", "fmov d2,x1", "fmov s2,w1"),
        }
        for ctor, (gg, ff, bd, bs) in expect.items():
            g = make(ctor)
            with self.subTest(isa=ctor.__name__):
                self.assertEqual(gg, g.move(dregs=[g.greg(1), g.greg(2)], dts=[U, U]).strip())
                self.assertEqual(ff, g.move(dregs=[g.freg(1, adt.DOUBLE), g.freg(2, adt.DOUBLE)],
                                            dts=[adt.DOUBLE, adt.DOUBLE]).strip())
                self.assertEqual(bd, g.move(dregs=[g.greg(1), g.freg(2, adt.DOUBLE)],
                                            dts=[adt.UINT64, adt.DOUBLE], modifiers=BITS).strip())
                self.assertEqual(bs, g.move(dregs=[g.greg(1), g.freg(2, adt.SINGLE)],
                                            dts=[adt.UINT32, adt.SINGLE], modifiers=BITS).strip())
        a = make(avx512)
        self.assertEqual(f"vmovq {a.rpref(a.greg(1))},{a.rpref(a.freg(2, adt.DOUBLE))}",
                         a.move(dregs=[a.greg(1), a.freg(2, adt.DOUBLE)], dts=[adt.UINT64, adt.DOUBLE],
                                modifiers=BITS).strip())

    def test_bits_pairs_sizes(self):
        g = make(neon)
        with self.assertRaises(ValueError):          # 4 bytes of bits into an 8-byte register
            g.move(dregs=[g.greg(1), g.freg(2, adt.DOUBLE)], dts=[adt.UINT32, adt.DOUBLE], modifiers=BITS)
        with self.assertRaises(ValueError):          # without BITS: no conversion between files
            g.move(dregs=[g.greg(1), g.freg(2, adt.DOUBLE)], dts=[adt.UINT64, adt.DOUBLE])

    def test_old_methods_on_top(self):
        for ctor in (rvv, neon, sve, avx512):
            g = make(ctor)
            with self.subTest(isa=ctor.__name__):
                self.assertEqual(g.move(dregs=[g.greg(1), g.greg(2)], dts=[U, U]),
                                 g.mov_greg(src=g.greg(1), dst=g.greg(2)))
                f1, f2 = g.freg(1, adt.SINGLE), g.freg(2, adt.SINGLE)
                self.assertEqual(g.move(dregs=[f1, f2], dts=[adt.SINGLE, adt.SINGLE]),
                                 g.mov_freg(src=f1, dst=f2, dt=adt.SINGLE))
                self.assertEqual(g.move(dregs=[g.greg(1), f2], dts=[adt.UINT32, adt.SINGLE], modifiers=BITS),
                                 g.mov_greg_to_freg(src=g.greg(1), dst=f2, dt=adt.SINGLE))

    def test_vector_moves_beside(self):
        for ctor in (rvv, neon, sve, sme, avx512):
            g = make(ctor)
            kinds = {tuple(sorted(o.rtype.name for o in s.operands.values() if o.rtype is not None))
                     for s in g.move.get_signatures()}
            with self.subTest(isa=ctor.__name__):
                self.assertIn(("GP", "GP"), kinds)
                self.assertTrue(any("VEC" in k for k in kinds))


if __name__ == "__main__":
    unittest.main()
