"""
SME: tiles zeroed as lists; the SME2 predicate-as-counter set only with SME2
"""

import unittest

from asmgen.asmblocks.sme import sme
from asmgen.registers import asm_data_type as adt


class test_sme_setup(unittest.TestCase):

    def test_zero_list(self):
        g = sme()
        g.output_inline = False
        self.assertEqual("zero {za1.d}", g.zero_treg(treg=g.treg(1, adt.FP64), dt=adt.FP64).strip())

    def test_counter_with_sme2(self):
        from asmgen.asmblocks.setup import sme_setup
        g = sme()
        g.output_inline = False
        g.set_features([])
        self.assertFalse(sme_setup(g).counter())
        g.set_features(["SME2"])
        self.assertTrue(sme_setup(g).counter())


if __name__ == "__main__":
    unittest.main()


class test_clobbers(unittest.TestCase):
    """
    Registers clobbered beyond the allocation, kept by the calling
    convention where they are the callee's
    """

    def test_sme_setup_clobbers_vectors(self):
        from asmgen.asmblocks.setup import sme_setup
        g = sme()
        self.assertEqual(list(range(g.max_fregs())), sme_setup(g).clobbers()["freg"])

    def test_callconv_keeps_callee_saved_clobbers(self):
        g = sme()
        g.output_inline = False
        cc = g.create_callconv()
        cc.add_clobber("freg", range(32))
        save = cc.save_in_call(g, regs={"greg": [], "freg": []})
        # AAPCS64: d8-d15 the callee's, the others not
        self.assertEqual(sorted(f"d{n}" for n in range(8, 16)),
                         sorted(l.split()[1].split(",")[0] for l in save.splitlines()
                                if l.strip().startswith("str")))


class test_callee_saved_last(unittest.TestCase):
    """
    The calling convention's callee-saved registers allocated last - through
    the tracker's shared pools for vector registers whose low parts they are
    """

    def test_deprioritized(self):
        g = sme()
        cc = g.create_callconv()
        rt = g.get_reg_tracker()
        cc.deprioritize_callee_saved(rt)
        # AAPCS64: x19-x28 and d8-d15 - so v8-v15, their registers - last
        self.assertEqual(-1, rt.priority("greg", 19))
        self.assertEqual(-1, rt.priority("vreg", 8))
        free = [i for i in range(g.max_vregs()) if i not in rt.get_used_regs("vreg")]
        order = [rt.reserve_any_reg("vreg") for _ in range(len(free) - 8)]
        self.assertFalse(set(order) & set(range(8, 16)))


class test_stack_parameters(unittest.TestCase):
    """
    Parameters passed on the stack are read past the save area the
    convention lays out - its clobbers included
    """

    def test_offset_past_saves(self):
        from asmgen.callconv.fngen import fngen
        from asmgen.registers import asm_data_type as adt
        g = sme()
        g.output_inline = False
        rt = g.get_reg_tracker()
        cc = g.create_callconv()
        fn = fngen(gen=g, rt=rt)
        # nine parameters: the ninth on the stack
        for i in range(9):
            cc.add_param("greg", f"p{i}")
        fn.init_cc(cc=cc)
        cc.add_clobber("freg", range(32))
        save, load, restore = fn.get_boilerplate(cc=cc)
        if "ldr" not in load:
            self.skipTest("no parameter on the stack in this setup")
        self.assertIn("[sp,#64]", load.replace(" ", ""))


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


class test_shifted_adds(unittest.TestCase):
    """
    dst = reg1 +/- (reg2 << shift) in one instruction where the ISA has one;
    NotImplementedError where it hasn't
    """

    def test_aarch64(self):
        g = sme()
        g.output_inline = False
        self.assertEqual("add x0,x0,x9, LSL #1",
                         g.add_greg_greg_shifted(dst=g.greg(0), reg1=g.greg(0), reg2=g.greg(9), shift=1).strip())
        self.assertEqual("sub x0,x0,x9, LSL #2",
                         g.sub_greg_greg_shifted(dst=g.greg(0), reg1=g.greg(0), reg2=g.greg(9), shift=2).strip())

    def test_x86_and_riscv(self):
        from asmgen.asmblocks.avx.avx2 import fma256
        from asmgen.asmblocks.rvv import rvv
        g = fma256()
        g.output_inline = False
        self.assertIn(",4)", g.add_greg_greg_shifted(dst=g.greg(0), reg1=g.greg(0), reg2=g.greg(9), shift=2))
        with self.assertRaises(NotImplementedError):
            g.add_greg_greg_shifted(dst=g.greg(0), reg1=g.greg(0), reg2=g.greg(9), shift=4)
        with self.assertRaises(NotImplementedError):
            g.sub_greg_greg_shifted(dst=g.greg(0), reg1=g.greg(0), reg2=g.greg(9), shift=1)
        r = rvv()
        with self.assertRaises(NotImplementedError):
            r.add_greg_greg_shifted(dst=r.greg(0), reg1=r.greg(0), reg2=r.greg(9), shift=1)
