"""
What each ISA sets up before a kernel and tears down after
"""

import unittest

from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.sme import sme
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.avx import avx512
from asmgen.asmblocks.setup import setup_kind
from asmgen.registers import reg_tracker, asm_data_type as adt


def tracker():
    return reg_tracker([("greg", 32), ("vreg", 32), ("freg", 32)])


def kinds(gen):
    return [s.kind for s in gen.isa_setup().steps()]


class test_setup(unittest.TestCase):

    def test_neon_needs_nothing(self):
        gen = neon()
        self.assertEqual([], kinds(gen))
        self.assertEqual("", gen.isa_setup().setup(rt=tracker(), dt=adt.FP64))
        with self.assertRaises(ValueError):
            gen.isa_setup().setup(rt=tracker(), dt=adt.FP64, vl=2)

    def test_sve_predicate(self):
        gen = sve()
        self.assertEqual([setup_kind.PREDICATE], kinds(gen))
        text = gen.isa_setup().setup(rt=tracker(), dt=adt.FP64)
        self.assertIn("ptrue p0.d", text)
        self.assertEqual("p0", str(gen.isa_setup().all_true_predicate(rt=tracker(), dt=adt.FP64)))
        self.assertFalse(gen.isa_setup().supports_vl_override())

    def test_sme_engine(self):
        gen = sme()
        setup = gen.isa_setup()
        self.assertEqual(setup_kind.ENGINE, kinds(gen)[0])
        text = setup.setup(rt=tracker(), dt=adt.FP64)
        self.assertIn("smstart", text.splitlines()[0])
        self.assertIn("ptrue p0.d", text)
        self.assertIn("pn8", text)
        self.assertIn("smstop", setup.teardown(rt=tracker(), dt=adt.FP64))

    def test_rvv_vector_length(self):
        gen = rvv()
        setup = gen.isa_setup()
        self.assertEqual([setup_kind.VECTOR_LENGTH], kinds(gen))
        self.assertTrue(setup.supports_vl_override())
        rt = tracker()
        full = setup.setup(rt=rt, dt=adt.FP64)
        self.assertIn("vsetvli", full)
        self.assertIn("zero", full)                 # VLMAX
        vlreg = setup.vector_length_register(rt=rt)
        self.assertIn(str(vlreg), full)
        # fewer elements than the hardware has: the length goes through avl
        rt = tracker()
        short = setup.setup(rt=rt, dt=adt.FP64, vl=4)
        avl = gen.greg(rt.aliased_regs["greg"]["avl"])
        self.assertIn(str(avl), short)
        self.assertIn("4", short)
        self.assertNotIn("zero", short.split("vsetvli")[1])

    def test_avx512_constants_in_their_own_section(self):
        gen = avx512()
        self.assertEqual("", gen.isadata())
        gen.ensure_indices(adt.FP64, 8)
        data = gen.isadata()
        lines = [l.strip().strip('"').replace("\\n", "").replace("\\t", "").strip()
                 for l in data.splitlines() if l.strip()]
        self.assertTrue(lines[0].startswith(".pushsection .rodata"), lines)
        self.assertTrue(lines[-1].startswith(".popsection"), lines)
        gen.isaclear()
        self.assertEqual("", gen.isadata())

    def test_old_wrappers(self):
        for gen in (neon(), sve(), sme(), rvv()):
            with self.subTest(gen=type(gen).__name__):
                self.assertEqual(gen.isa_setup().setup(rt=tracker(), dt=adt.FP32),
                                 gen.isaquirks(rt=tracker(), dt=adt.FP32))
                self.assertEqual(gen.isa_setup().teardown(rt=tracker(), dt=adt.FP32),
                                 gen.isaendquirks(rt=tracker(), dt=adt.FP32))


if __name__ == "__main__":
    unittest.main()
