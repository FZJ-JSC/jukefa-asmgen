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
        kinds = lambda g: {o.rtype.name for s in g.set.get_signatures() for o in s.operands.values() if o.rtype is not None}
        self.assertIn("TILE", kinds(make(sme)))
        self.assertNotIn("TILE", kinds(make(sve)))
        self.assertTrue({"GP", "FP", "VEC"} <= kinds(make(rvv)))


if __name__ == "__main__":
    unittest.main()


class test_one_and_values(unittest.TestCase):
    """
    CONST one (1, 1.0 in dt) and values in general registers
    """
    def one(self, g, reg, dt):
        return g.set(dreg=reg, dt=dt, modifiers=CONST, constant="one").strip().replace("\n", " ; ")

    def value(self, g, v):
        return g.set(dreg=g.greg(1), dt=adt.UINT64, modifiers=CONST, value=v).strip().replace("\n", " ; ")

    def test_one(self):
        r, n, s, a = make(rvv), make(neon), make(sve), make(avx512)
        self.assertEqual("vmv.v.i v1,1", self.one(r, r.vreg(1), adt.SINT32))
        self.assertEqual("fmov v1.2d,#1.0", self.one(n, n.vreg(1), adt.DOUBLE))
        self.assertEqual("movi v1.8h,#1", self.one(n, n.vreg(1), adt.UINT16))
        self.assertEqual("fdup z1.s,#1.0", self.one(s, s.vreg(1), adt.SINGLE))
        self.assertEqual("dup z1.d,#1", self.one(s, s.vreg(1), adt.SINT64))
        self.assertEqual("fmov d1,#1.0", self.one(n, n.freg(1, adt.DOUBLE), adt.DOUBLE))
        self.assertIn("vpsllq $54", self.one(a, a.vreg(1), adt.DOUBLE))
        self.assertIn("vpabsb", self.one(make(fma256), make(fma256).vreg(1), adt.SINT8))
        self.assertEqual("li t1,1", self.one(r, r.greg(1), adt.UINT64))
        for g, reg, dt in ((r, r.vreg(1), adt.DOUBLE), (r, r.freg(1, adt.DOUBLE), adt.DOUBLE),
                           (n, n.vreg(1), adt.SINT64)):
            with self.subTest(isa=type(g).__name__, dt=dt.name), self.assertRaises(ValueError):
                self.one(g, reg, dt)              # takes a scratch register: not offered

    def test_values(self):
        n, r, a = make(neon), make(rvv), make(avx512)
        self.assertEqual("mov x1,#6148914691236517205", self.value(n, 0x5555555555555555))   # a bitmask
        self.assertEqual("movz x1,#48879 ; movk x1,#57005, lsl #16", self.value(n, 0xDEADBEEF))
        self.assertEqual(f"li {r.greg(1)},81985529216486895", self.value(r, 0x0123456789ABCDEF))
        p = a.rpref(a.greg(1))
        self.assertEqual(f"movq $-1,{p}", self.value(a, 2**64 - 1))
        self.assertEqual(f"movq $4294967296,{p}", self.value(a, 2**32))       # assembled as movabs
        with self.assertRaises(ValueError):
            self.value(n, 2**64)

    def test_mov_greg_imm_on_set(self):
        for ctor in (rvv, neon, avx512):
            g = make(ctor)
            with self.subTest(isa=ctor.__name__):
                self.assertEqual(g.set(dreg=g.greg(1), dt=adt.UINT64, modifiers=CONST, value=4096),
                                 g.mov_greg_imm(reg=g.greg(1), imm=4096))
