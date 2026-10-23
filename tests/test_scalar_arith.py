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
from asmgen.asmblocks.op.operand import operand_modifier as opd_mod
from asmgen.asmblocks.op.opd3 import opd3_modifier

DIR = {'shl': {opd3_modifier.SHL}, 'shr': {opd3_modifier.SHR}}

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
        return plain(getattr(g, op)(adreg=g.greg(a), bimm=imm, cdreg=g.greg(c), a_dt=U, c_dt=U))

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
                self.assertEqual(g.add(adreg=a, bimm=16, cdreg=a, a_dt=U, c_dt=U), g.add_greg_imm(reg=a, imm=16))
                self.assertEqual(g.sub(adreg=a, bdreg=b, cdreg=c, a_dt=U, b_dt=U, c_dt=U),
                                 g.sub_greg_greg(dst=c, reg1=a, reg2=b))

    def test_vector_and_scalar_behind_one_name(self):
        g = make(rvv)
        names = [{n for n in s.operands} for s in g.add.get_signatures()]
        self.assertIn({"adreg", "cdreg", "bimm"}, names)
        v = g.vreg
        self.assertIn("vfadd.vv", g.add(adreg=v(1), bdreg=v(2), cdreg=v(3), a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64))
        self.assertTrue(hasattr(g.add, "get_base_inst"))          # the vector operation's own


class test_shifts(unittest.TestCase):
    """
    shift (SHL, SHR); shifted operands (operand modifier ISHIFT, bdreg_shift)
    """
    def sh(self, g, op, a, imm, c):
        return plain(g.shift(adreg=g.greg(a), bimm=imm, cdreg=g.greg(c), a_dt=U, c_dt=U, modifiers=DIR[op]))

    def shifted(self, g, op, a, b, c, s):
        return plain(getattr(g, op)(adreg=g.greg(a), bdreg=g.greg(b), cdreg=g.greg(c), a_dt=U, b_dt=U, c_dt=U,
                                    operand_modifiers={"bdreg": {opd_mod.ISHIFT}}, bdreg_shift=s))

    def test_shl_shr(self):
        r = make(rvv)
        x = lambda i: str(r.greg(i))
        self.assertEqual(f"slli {x(3)},{x(1)},63", self.sh(r, "shl", 1, 63, 3))
        self.assertEqual(f"srli {x(3)},{x(1)},4", self.sh(r, "shr", 1, 4, 3))
        self.assertEqual(f"sll {x(3)},{x(1)},{x(2)}",
                         plain(r.shift(adreg=r.greg(1), bdreg=r.greg(2), cdreg=r.greg(3), a_dt=U, b_dt=U, c_dt=U, modifiers=DIR['shl'])))
        n = make(neon)
        self.assertEqual("lsl x3,x1,#5", self.sh(n, "shl", 1, 5, 3))
        self.assertEqual("lsr x3,x1,x2",
                         plain(n.shift(adreg=n.greg(1), bdreg=n.greg(2), cdreg=n.greg(3), a_dt=U, b_dt=U, c_dt=U, modifiers=DIR['shr'])))
        a = make(avx512)
        p = lambda i: a.rpref(a.greg(i))
        self.assertEqual(f"shlq $3,{p(1)}", self.sh(a, "shl", 1, 3, 1))
        self.assertEqual(f"movq {p(1)},{p(3)} ; shrq $3,{p(3)}", self.sh(a, "shr", 1, 3, 3))
        with self.assertRaises(ValueError):      # an amount in a register: only %cl, not offered
            a.shift(adreg=a.greg(1), bdreg=a.greg(2), cdreg=a.greg(1), a_dt=U, b_dt=U, c_dt=U, modifiers=DIR['shl'])
        for g in (r, n, a):
            with self.subTest(isa=type(g).__name__), self.assertRaises(ValueError):
                self.sh(g, "shl", 1, 64, 1)

    def test_shifted_operands(self):
        n = make(neon)
        self.assertEqual("add x3,x1,x2, LSL #3", self.shifted(n, "add", 1, 2, 3, 3))
        self.assertEqual("sub x3,x1,x2, LSL #63", self.shifted(n, "sub", 1, 2, 3, 63))
        self.assertEqual("add x3,x1,x2", self.shifted(n, "add", 1, 2, 3, 0))
        a = make(avx512)
        p = lambda i: a.rpref(a.greg(i))
        self.assertEqual(f"leaq ({p(1)},{p(2)},8),{p(3)}", self.shifted(a, "add", 1, 2, 3, 3))
        for op, s in (("add", 4), ("sub", 1)):           # lea: scales 1..8, no subtraction
            with self.subTest(op=op, shift=s), self.assertRaises(ValueError):
                self.shifted(a, op, 1, 2, 3, s)
        r = make(rvv)
        r.set_features([])
        with self.assertRaises(ValueError):              # Zba: not in the baseline
            self.shifted(r, "add", 1, 2, 3, 2)
        r.set_features(["ZBA"])
        x = lambda i: str(r.greg(i))
        self.assertEqual(f"sh2add {x(3)},{x(2)},{x(1)}", self.shifted(r, "add", 1, 2, 3, 2))
        for s in (0, 4):
            with self.subTest(shift=s), self.assertRaises(ValueError):
                self.shifted(r, "add", 1, 2, 3, s)

    def test_old_methods_on_top(self):
        for ctor in (rvv, neon, avx512):
            g = make(ctor)
            r = g.greg(1)
            with self.subTest(isa=ctor.__name__):
                self.assertEqual(g.shift(adreg=r, bimm=3, cdreg=r, a_dt=U, c_dt=U, modifiers=DIR['shl']), g.shift_greg_left(reg=r, bit_count=3))
                self.assertEqual(g.shift(adreg=r, bimm=3, cdreg=r, a_dt=U, c_dt=U, modifiers=DIR['shr']), g.shift_greg_right(reg=r, bit_count=3))
        a = make(avx512)
        with self.assertRaises(NotImplementedError):     # what ukrgen's probe catches
            a.sub_greg_greg_shifted(dst=a.greg(1), reg1=a.greg(1), reg2=a.greg(2), shift=1)

class test_vector_lengths(unittest.TestCase):
    """
    VOFFSET: bimm counts vector lengths (SVE)
    """
    def vl(self, g, a, n, c):
        return plain(g.add(adreg=g.greg(a), bimm=n, cdreg=g.greg(c), a_dt=U, c_dt=U,
                           modifiers={opd3_modifier.VOFFSET}))

    def test_sve(self):
        g = make(sve)
        self.assertEqual("incb x1, ALL, MUL #3", self.vl(g, 1, 3, 1))
        self.assertEqual("decb x1, ALL, MUL #16", self.vl(g, 1, -16, 1))
        self.assertEqual("addvl x1,x1,#17", self.vl(g, 1, 17, 1))
        self.assertEqual("addvl x3,x1,#-32", self.vl(g, 1, -32, 3))
        for bad in (32, -33):
            with self.subTest(n=bad), self.assertRaises(ValueError):
                self.vl(g, 1, bad, 1)
        self.assertEqual("incb x1, ALL, MUL #16 ; incb x1, ALL, MUL #4",
                         plain(g.add_greg_voff(reg=g.greg(1), offset=20, dt=adt.FP64)))

    def test_fixed_lengths_count_bytes(self):
        n = make(neon)
        with self.assertRaises(ValueError):          # no VL-scaled form: a byte count instead
            self.vl(n, 1, 2, 1)
        self.assertEqual("add x1,x1,#32", plain(n.add_greg_voff(reg=n.greg(1), offset=2, dt=adt.FP64)))
        with self.assertRaises(ValueError):          # 2047 * 16 bytes: no add immediate
            n.add_greg_voff(reg=n.greg(1), offset=2047, dt=adt.FP64)

if __name__ == "__main__":
    unittest.main()
