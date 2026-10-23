"""
Scalar arithmetic as operations (docs/V06_REWORK.md, phase 2): add, sub and
mul on general registers, immediates within each ISA's range, the old
methods on top of them
"""
import unittest

from asmgen.asmblocks.avx import avx512
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.sve import sve
from asmgen.registers import asm_data_type as adt

U = adt.UINT64


def plain(text):
    return " ; ".join(l.strip() for l in text.strip().splitlines())


def make(ctor):
    g = ctor()
    g.set_output_inline(yesno=False)
    return g


class test_scalar_arith(unittest.TestCase):
    def rr(self, g, op, a, b, c):
        return plain(getattr(g, op)(adreg=g.greg(a), bdreg=g.greg(b), cdreg=g.greg(c), a_dt=U, b_dt=U, c_dt=U))

    def ri(self, g, op, a, imm, c):
        return plain(getattr(g, op)(adreg=g.greg(a), imm=imm, cdreg=g.greg(c), a_dt=U, c_dt=U))

    def test_riscv(self):
        g = make(rvv)
        r = lambda i: str(g.greg(i))
        self.assertEqual(f"add {r(3)},{r(1)},{r(2)}", self.rr(g, "add", 1, 2, 3))
        self.assertEqual(f"sub {r(3)},{r(1)},{r(2)}", self.rr(g, "sub", 1, 2, 3))
        self.assertEqual(f"mul {r(3)},{r(1)},{r(2)}", self.rr(g, "mul", 1, 2, 3))
        self.assertEqual(f"add {r(3)},{r(1)},2047", self.ri(g, "add", 1, 2047, 3))
        self.assertEqual(f"add {r(3)},{r(1)},-2048", self.ri(g, "add", 1, -2048, 3))
        self.assertEqual(f"add {r(3)},{r(1)},-2048", self.ri(g, "sub", 1, 2048, 3))
        for op, bad in (("add", 2048), ("add", -2049), ("sub", -2048), ("sub", 2049)):
            with self.subTest(op=op, imm=bad), self.assertRaises(ValueError):
                self.ri(g, op, 1, bad, 3)
        with self.assertRaises(ValueError):          # no immediate form
            self.ri(g, "mul", 1, 3, 3)

    def test_aarch64(self):
        for ctor in (neon, sve):
            g = make(ctor)
            self.assertEqual("add x3,x1,x2", self.rr(g, "add", 1, 2, 3))
            self.assertEqual("sub x3,x1,#4095", self.ri(g, "sub", 1, 4095, 3))
            self.assertEqual("add x3,x1,#-8", self.ri(g, "add", 1, -8, 3))
            self.assertEqual("add x3,x1,#8192", self.ri(g, "add", 1, 8192, 3))           # lsl #12
            self.assertEqual(f"add x3,x1,#{4095 * 4096}", self.ri(g, "add", 1, 4095 * 4096, 3))
            for bad in (4097, 5000, 4096 * 4096, -4097):
                with self.subTest(imm=bad), self.assertRaises(ValueError):
                    self.ri(g, "add", 1, bad, 3)
            with self.assertRaises(ValueError):
                self.ri(g, "mul", 1, 3, 3)

    def test_x86(self):
        g = make(avx512)
        p = lambda i: g.rpref(g.greg(i))
        self.assertEqual(f"movq {p(1)},{p(3)} ; addq {p(2)},{p(3)}", self.rr(g, "add", 1, 2, 3))
        self.assertEqual(f"addq {p(2)},{p(1)}", self.rr(g, "add", 1, 2, 1))
        # the destination is the second source: kept, not overwritten
        self.assertEqual(f"addq {p(1)},{p(2)}", self.rr(g, "add", 1, 2, 2))
        self.assertEqual(f"imulq {p(1)},{p(2)}", self.rr(g, "mul", 1, 2, 2))
        self.assertEqual(f"negq {p(2)} ; addq {p(1)},{p(2)}", self.rr(g, "sub", 1, 2, 2))
        self.assertEqual(f"imulq $12,{p(1)},{p(3)}", self.ri(g, "mul", 1, 12, 3))
        self.assertEqual(f"addq $-2147483648,{p(1)}", self.ri(g, "add", 1, -2**31, 1))
        with self.assertRaises(ValueError):
            self.ri(g, "add", 1, 2**31, 1)

    def test_old_methods_on_top(self):
        for ctor in (rvv, neon, avx512):
            g = make(ctor)
            a, b, c = g.greg(1), g.greg(2), g.greg(3)
            with self.subTest(isa=ctor.__name__):
                self.assertEqual(g.add(adreg=a, bdreg=b, cdreg=c, a_dt=U, b_dt=U, c_dt=U),
                                 g.add_greg_greg(dst=c, reg1=a, reg2=b))
                self.assertEqual(g.add(adreg=a, imm=16, cdreg=a, a_dt=U, c_dt=U), g.add_greg_imm(reg=a, imm=16))
                self.assertEqual(g.sub(adreg=a, bdreg=b, cdreg=c, a_dt=U, b_dt=U, c_dt=U),
                                 g.sub_greg_greg(dst=c, reg1=a, reg2=b))

    def test_vector_and_scalar_behind_one_name(self):
        g = make(rvv)
        names = [{n for n in s.operands} for s in g.add.get_signatures()]
        self.assertIn({"adreg", "cdreg", "imm"}, names)
        v = g.vreg
        self.assertIn("vfadd.vv", g.add(adreg=v(1), bdreg=v(2), cdreg=v(3), a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64))
        self.assertTrue(hasattr(g.add, "get_base_inst"))          # the vector operation's own


if __name__ == "__main__":
    unittest.main()
