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


from asmgen.asmblocks.setup import predication


class test_reg_trackers(unittest.TestCase):

    def test_reserved_never_allocated(self):
        cases = ((neon, {"sp", "x30", "x29", "x18"}),
                 (rvv, {"ra", "sp", "gp", "tp", "s0", "zero"}),
                 (avx512, {"rsp", "rbp"}))
        for G, reserved in cases:
            with self.subTest(isa=G.__name__):
                g = G()
                rt = g.get_reg_tracker()
                free = g.max_gregs() - len(rt.get_used_regs("greg"))
                handed = {str(g.greg(rt.reserve_any_reg("greg"))) for _ in range(free)}
                self.assertFalse(handed & reserved)

    def test_avx512_k0(self):
        self.assertEqual({"k0": 0}, avx512().get_reg_tracker().aliased_regs["mreg"])


class test_predication(unittest.TestCase):

    def test_kinds(self):
        self.assertEqual(predication.NONE, neon().isa_setup().predication())
        self.assertEqual(predication.EXPLICIT, sve().isa_setup().predication())
        self.assertEqual(predication.IMPLICIT, rvv().isa_setup().predication())

    def test_chosen_predicate_aliased(self):
        g = sve()
        rt = g.get_reg_tracker()
        setup = g.isa_setup(predicate=3)
        self.assertIn("ptrue p3.d", setup.setup(rt=rt, dt=adt.FP64))
        self.assertEqual({"all_true": 3}, rt.aliased_regs["mreg"])
        self.assertIn(3, rt.get_used_regs("mreg"))
        # later calls find it through the tracker
        self.assertEqual("p3", str(g.isa_setup().all_true_predicate(rt=rt, dt=adt.FP64)))


class test_shared_registers(unittest.TestCase):

    def test_fp_and_vector_are_one_file(self):
        g = neon()
        rt = g.get_reg_tracker()
        v = rt.reserve_any_reg("vreg")
        f = rt.reserve_any_reg("freg")
        self.assertNotEqual(v, f)
        self.assertEqual({v, f}, rt.get_used_regs("vreg"))
        rt.unuse_reg("freg", f)
        self.assertEqual({v}, rt.get_used_regs("vreg"))

    def test_rvv_mask_is_v0(self):
        g = rvv()
        rt = g.get_reg_tracker()
        order = [rt.reserve_any_reg("vreg") for _ in range(g.max_vregs())]
        self.assertEqual(0, order[-1])          # v0 comes last
        rt = g.get_reg_tracker()
        self.assertEqual(0, rt.reserve_any_reg("mreg"))
        self.assertIn(0, rt.get_used_regs("vreg"))


class test_rvv_index_vector(unittest.TestCase):

    def test_bytes(self):
        g = rvv()
        text = g.greg_to_voffs(streg=g.greg(5), vreg=g.vreg(3), dt=adt.FP64)
        self.assertIn("vid.v v3", text)
        self.assertIn("vmul.vx v3,v3,", text)


class test_element_state(unittest.TestCase):
    """
    Whether the element type is state, and switching it over register
    groups the caller counts - the ISA's own grouping kept
    """

    def test_rvv(self):
        from asmgen.asmblocks.rvv import rvv
        from asmgen.asmblocks.setup import rvv_setup
        g = rvv()
        g.output_inline = False
        self.assertTrue(rvv_setup(g).element_type_is_state())
        self.assertEqual("vsetvli zero,zero,e32,m2,ta,ma",
                         rvv_setup(g).setup_dt(dt=adt.FP32, group=2).strip())

    def test_per_instruction(self):
        from asmgen.asmblocks.setup import sme_setup
        self.assertFalse(sme_setup(sme()).element_type_is_state())
        self.assertEqual("", sme_setup(sme()).setup_dt(dt=adt.FP32, group=2))


if __name__ == "__main__":
    unittest.main()
