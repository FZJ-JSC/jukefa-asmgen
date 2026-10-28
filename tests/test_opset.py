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

class test_masks(unittest.TestCase):
    """
    CONSTMASK: mask registers all true or all false
    """
    def mask(self, g, reg, dt, mask):
        return g.set(dreg=reg, dt=dt, modifiers={opset_modifier.CONSTMASK}, mask=mask).strip()

    def test_sve_sme(self):
        for ctor in (sve, sme):
            g = make(ctor)
            with self.subTest(isa=ctor.__name__):
                self.assertEqual("ptrue p1.d", self.mask(g, g.mreg(1), adt.DOUBLE, "alltrue"))
                self.assertEqual("pfalse p1.b", self.mask(g, g.mreg(1), adt.SINGLE, "allfalse"))
                self.assertEqual("ptrue pn8.s", self.mask(g, g.mreg(8, is_pn=True), adt.SINGLE, "alltrue"))
                self.assertEqual(g.set(dreg=g.mreg(2), dt=adt.HALF, modifiers={opset_modifier.CONSTMASK}, mask="alltrue"),
                                 g.set(dreg=g.mreg(2), dt=adt.HALF, modifiers={opset_modifier.CONSTMASK}, mask="alltrue"))

    def test_avx512(self):
        g = make(avx512)
        self.assertEqual("kxnorw %k1,%k1,%k1", self.mask(g, g.mreg(1), adt.DOUBLE, "alltrue"))
        self.assertEqual("kxorw %k1,%k1,%k1", self.mask(g, g.mreg(1), adt.DOUBLE, "allfalse"))
        self.assertEqual(g.set(dreg=g.mreg(1), dt=adt.SINGLE, modifiers={opset_modifier.CONSTMASK}, mask="alltrue"),
                         g.set(dreg=g.mreg(1), dt=adt.SINGLE, modifiers={opset_modifier.CONSTMASK}, mask="alltrue"))

    def test_avx2_vector_masks(self):
        from asmgen.asmblocks.avx import fma128
        for ctor in (fma128, fma256):
            g = make(ctor)
            v = g.vreg(3)
            p = g.rpref(v)
            with self.subTest(isa=ctor.__name__):
                self.assertEqual(f"vpcmpeqd {p},{p},{p}", self.mask(g, v, adt.DOUBLE, "alltrue"))
                self.assertEqual(f"vpxor {p},{p},{p}", self.mask(g, v, adt.DOUBLE, "allfalse"))
                self.assertEqual(f"vpcmpeqd {p},{p},{p}", g.set(dreg=v, dt=adt.SINGLE, modifiers={opset_modifier.CONSTMASK}, mask="alltrue").strip())
        g = make(avx512)
        with self.assertRaises(ValueError):              # AVX-512's masks are k registers
            self.mask(g, g.vreg(3), adt.DOUBLE, "alltrue")

    def test_no_masks_no_forms(self):
        for ctor in (rvv, neon):
            kinds = {o.rtype.name for s in make(ctor).set.get_signatures() for o in s.operands.values() if o.rtype}
            with self.subTest(isa=ctor.__name__):
                self.assertNotIn("MASK", kinds)
        with self.assertRaises(ValueError):
            g = make(sve)
            self.mask(g, g.mreg(1), adt.DOUBLE, "sometrue")


class test_vlen(unittest.TestCase):
    """
    VLEN: the vector length into a general register - bytes, or elements of
    dt; simd_size_to_greg, vlen_to_greg and c_simd_size_function on it
    """
    def vlen(self, g, dt=None):
        return g.set(dreg=g.greg(1), dt=dt, modifiers={opset_modifier.VLEN}).strip()

    def test_every_backend(self):
        from asmgen.asmblocks.rvv071 import rvv071
        r, n, s, a = make(rvv), make(neon), make(sve), make(avx512)
        x = str(r.greg(1))
        self.assertEqual(f"vsetvli {x}, zero, e8, m1, ta, ma", self.vlen(r))
        self.assertEqual(f"vsetvli {x}, zero, e64, m1, ta, ma", self.vlen(r, adt.DOUBLE))
        r.set_parameter(name="LMUL", value=2)
        self.assertEqual(f"vsetvli {x}, zero, e32, m2, ta, ma", self.vlen(r, adt.SINGLE))
        self.assertEqual(f"th.vsetvli {x}, zero, e16, m1", self.vlen(make(rvv071), adt.HALF))   # XTheadVector, the default
        self.assertEqual("mov x1,#16", self.vlen(n))
        self.assertEqual("mov x1,#4", self.vlen(n, adt.SINGLE))
        self.assertEqual("cntb x1", self.vlen(s))
        self.assertEqual("cntd x1", self.vlen(s, adt.DOUBLE))
        self.assertEqual(f"movq $8,{a.rpref(a.greg(1))}", self.vlen(a, adt.DOUBLE))

    def test_c_helper_asm(self):
        c = make(sme).c_simd_size_function
        self.assertIn('"smstart', c)
        self.assertIn('"cntb %[byte_size]', c)
        self.assertIn('"smstop', c)
        self.assertEqual("static inline size_t get_simd_size() { return 64; }", make(avx512).c_simd_size_function)


class test_index(unittest.TestCase):
    """
    STINDEX: element i of a vector register is i times a stride register;
    greg_to_voffs on it
    """
    def index(self, g, dt):
        return g.set(dreg=g.vreg(2), dt=dt, modifiers={opset_modifier.STINDEX}, streg=g.greg(1)).strip().replace("\n", " ; ")

    def test_backends(self):
        r, s = make(rvv), make(sve)
        self.assertEqual(f"vid.v v2 ; vmul.vx v2,v2,{r.greg(1)}", self.index(r, adt.DOUBLE))
        self.assertEqual("index z2.d,#0,x1", self.index(s, adt.DOUBLE))
        self.assertEqual("index z2.s,#0,w1", self.index(s, adt.SINGLE))
        a = make(avx512)
        text = self.index(a, adt.DOUBLE)
        self.assertIn("vpmuludq", text)
        self.assertTrue(any("iota_64x8" in str(k) for k in a.asmdata), "the index table registered")

    def test_where_there_is_none(self):
        n = make(neon)
        with self.assertRaises(ValueError):
            self.index(n, adt.DOUBLE)
        with self.assertRaises(ValueError):
            n.set(dreg=n.vreg(2), dt=adt.DOUBLE, modifiers={opset_modifier.STINDEX}, streg=n.greg(1))
        with self.assertRaises(ValueError):              # no byte multiply on AVX
            self.index(make(avx512), adt.SINT8)

class test_avx2_masks(unittest.TestCase):
    """
    AVX2's masks are vector registers: mreg(i) is vreg(i), the tracker
    shares their pool; AVX-512's are k registers, k0 reserved
    """
    def test_avx2(self):
        from asmgen.asmblocks.avx import fma128
        for ctor in (fma128, fma256):
            g = make(ctor)
            with self.subTest(isa=ctor.__name__):
                self.assertEqual(str(g.vreg(3)), str(g.mreg(3)))
                rt = g.get_reg_tracker()
                m = rt.reserve_any_reg("mreg")
                self.assertNotEqual(m, rt.reserve_any_reg("vreg"))     # one pool
                p = g.rpref(g.mreg(2))
                self.assertEqual(f"vpcmpeqd {p},{p},{p}", g.set(dreg=g.mreg(2), dt=adt.DOUBLE, modifiers={opset_modifier.CONSTMASK}, mask="alltrue").strip())
                kinds = {o.rtype.name for s in g.set.get_signatures() for o in s.operands.values() if o.rtype}
                self.assertNotIn("MASK", kinds)                        # no k registers on AVX2

    def test_avx512(self):
        g = make(avx512)
        self.assertEqual("k3", str(g.mreg(3)))
        self.assertEqual({"k0": 0}, g.reserved_registers()["mreg"])
        self.assertNotIn(("mreg", "vreg", None), g.shared_registers())


class test_param(unittest.TestCase):
    """
    PARAM: set's destination an inline asm register operand %[param], no
    dreg - c_simd_size_function's
    """
    def test_vlen_into_a_param(self):
        from asmgen.asmblocks.op.opset import opset_modifier as M
        for ctor, text in ((rvv, "vsetvli %[n], zero, e8, m1, ta, ma"), (sve, "cntb %[n]"), (neon, "mov %[n],#16")):
            g = make(ctor)
            with self.subTest(isa=ctor.__name__):
                self.assertEqual(text, g.set(modifiers={M.VLEN, M.PARAM}, param="n").strip())
        g = make(sve)
        self.assertEqual("cntd %[n]", g.set(dt=adt.DOUBLE, modifiers={M.VLEN, M.PARAM}, param="n").strip())
        self.assertIn('"cntb %[byte_size]', make(sme).c_simd_size_function)

    def test_dt_needed_but_for_vlen(self):
        g = make(rvv)
        with self.assertRaises(ValueError):
            g.set(dreg=g.vreg(1), modifiers=CONST, constant="zero")


def plain(text):
    """an instruction or several, one line each, as one: separated by ;"""
    lines = [l.strip().strip('"').replace("\\n", "").replace("\\t", "").strip() for l in text.split("\n")]
    return " ; ".join(l for l in lines if l)


class test_first_masks(unittest.TestCase):
    """
    CONSTMASK mask="first": the first count elements active - a vector
    along cdim under it packs fewer elements than a vector has
    """
    def first(self, g, reg, dt, count):
        return dict(dreg=reg, dt=dt, modifiers={opset_modifier.CONSTMASK}, mask="first", count=count)

    def test_sve_patterns(self):
        g = make(sve)
        self.assertEqual("ptrue p1.d, vl4", plain(g.set(**self.first(g, g.mreg(1), adt.DOUBLE, 4))))
        self.assertEqual("ptrue p1.s, vl16", plain(g.set(**self.first(g, g.mreg(1), adt.SINGLE, 16))))
        self.assertIsNone(g.set.fits(**self.first(g, g.mreg(1), adt.DOUBLE, 9)))    # not a pattern

    def test_no_counter_predicates(self):
        from asmgen.asmblocks.sme import sme
        g = make(sme)
        self.assertIsNone(g.set.fits(**self.first(g, g.mreg(8, True), adt.DOUBLE, 4)))

    def test_avx512_shifted(self):
        g = make(avx512)
        self.assertEqual(["kxnorw %k1,%k1,%k1", "kshiftrw $12,%k1,%k1"],
                         plain(g.set(**self.first(g, g.mreg(1), adt.SINGLE, 4))).split(" ; "))
        self.assertIsNone(g.set.fits(**self.first(g, g.mreg(1), adt.DOUBLE, 9)))     # 8 lanes

    def test_none_elsewhere(self):
        for gen in (fma256, rvv, neon):
            g = make(gen)
            with self.subTest(gen=gen.__name__):
                self.assertFalse([s for s in g.set.forms(modifiers={opset_modifier.CONSTMASK})
                                  if "first" == s.structural_params.get("mask")])


class test_avl(unittest.TestCase):
    """
    The first avl elements from a register: SVE's mask of them (MASKAVL,
    whilelo - any count, at most the vector's), RVV's vector length for them
    (VLEN with avlreg, vsetvli rd, avl)
    """
    def test_sve_whilelo(self):
        g = make(sve)
        self.assertEqual("whilelo p1.d, xzr, x3", plain(g.set(dreg=g.mreg(1), dt=adt.DOUBLE,
                                                             modifiers={opset_modifier.MASKAVL}, avlreg=g.greg(3))))
        self.assertIsNone(g.set.fits(dreg=g.greg(2), dt=adt.DOUBLE, modifiers={opset_modifier.VLEN},
                                     avlreg=g.greg(3)))     # no vector length to set

    def test_no_counter_predicates(self):
        from asmgen.asmblocks.sme import sme
        g = make(sme)
        self.assertIsNone(g.set.fits(dreg=g.mreg(8, True), dt=adt.DOUBLE, modifiers={opset_modifier.MASKAVL},
                                     avlreg=g.greg(3)))

    def test_rvv_vsetvli(self):
        g = make(rvv)
        self.assertEqual("vsetvli t2, t3, e64, m1, ta, ma",
                         plain(g.set(dreg=g.greg(2), dt=adt.DOUBLE, modifiers={opset_modifier.VLEN}, avlreg=g.greg(3))))
        self.assertEqual("vsetvli t2, zero, e64, m1, ta, ma",
                         plain(g.set(dreg=g.greg(2), dt=adt.DOUBLE, modifiers={opset_modifier.VLEN})))

    def test_rvv_takes_a_scratch_vector(self):
        # RVV's: the element indices (vid.v) in a scratch vector, below the
        # count (vmsltu.vx) - masked accesses take it in v0
        forms = make(rvv).set.forms(modifiers={opset_modifier.MASKAVL})
        self.assertTrue(forms)
        self.assertTrue(all("tmpreg" in s.operands for s in forms))

    def test_none_elsewhere(self):
        # AVX-512's takes a scratch register (its own patch)
        for gen in (avx512, neon):
            g = make(gen)
            with self.subTest(gen=gen.__name__):
                self.assertFalse(g.set.forms(modifiers={opset_modifier.MASKAVL}))

