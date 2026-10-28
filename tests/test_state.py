"""
Processor state in signatures: what each instruction requires and writes
(state_requires, state_writes), declared by its generator - RVV's vector
length and type, SME's streaming mode and ZA (FA64 deciding the gathers') -
and what each ISA's setup and teardown leave
"""
import unittest

from asmgen.asmblocks.op.opmem import opmem_modifier as M
from asmgen.asmblocks.op.opset import opset_modifier as S
from asmgen.asmblocks.op.operand import register_type as rt
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.sme import sme
from asmgen.asmblocks.sve import sve
from asmgen.registers import asm_data_type as adt

SET = {"vlen": "!=None", "vtype": "!=None"}


class test_rvv(unittest.TestCase):
    def test_vector_forms_require_vlen(self):
        g = rvv()
        for op in (g.load, g.store):
            with self.subTest(op=op):
                self.assertTrue(all(SET == s.state_requires for s in op.forms(rtypes={"adreg": rt.VEC})))
                self.assertTrue(all({} == s.state_requires for s in op.forms(rtypes={"adreg": rt.GP})))
        self.assertEqual(frozenset({"vlen", "vtype"}),
                         next(iter(g.load.forms(rtypes={"adreg": rt.VEC}))).state_reads)

    def test_vsetvli_writes(self):
        writes = {tuple(sorted(f.operands)): f.state_writes for f in rvv().set.forms(modifiers={S.VLEN})}
        self.assertEqual({"vlen": "avl", "vtype": "dt"}, writes[("avlreg", "dreg")])
        self.assertEqual({"vlen": "max", "vtype": "dt"}, writes[("dreg",)])

    def test_kept_across_lmul(self):
        g = rvv()
        g.set_parameter("LMUL", 2)
        self.assertTrue(all(SET == s.state_requires for s in g.store.forms(rtypes={"adreg": rt.VEC})))

    def test_setup(self):
        self.assertEqual({"vlen": "max", "vtype": "e64"}, rvv().isa_setup().state(adt.FP64))


class test_sme(unittest.TestCase):
    def test_tiles_need_streaming_and_za(self):
        g = sme()
        tiles = [f for f in g.load.forms() if any(rt.TILE == o.rtype for o in f.operands.values())]
        self.assertTrue(tiles)
        self.assertTrue(all({"sm": "==1", "za": "==1"} == f.state_requires for f in tiles))

    def test_gathers_and_fa64(self):
        # streaming mode runs gathers and scatters only with FEAT_SME_FA64
        # (LX2 has it, Apple's parts don't)
        g = sme()
        gathers = lambda: list(g.load.forms(modifiers={M.VINDEX})) + list(g.store.forms(modifiers={M.VINDEX}))
        self.assertTrue(all({} == f.state_requires for f in gathers()))
        g.set_parameter("FA64", "off")
        self.assertTrue(gathers() and all({"sm": "==0"} == f.state_requires for f in gathers()))
        g.set_parameter("FA64", "on")
        self.assertTrue(all({} == f.state_requires for f in gathers()))

    def test_setup(self):
        setup = sme().isa_setup()
        self.assertEqual({"sm": "1", "za": "1"}, setup.state(adt.FP64))
        self.assertEqual({"sm": "0", "za": "0"}, setup.teardown_state())


class test_no_state(unittest.TestCase):
    def test_sve(self):
        g = sve()
        self.assertTrue(all({} == f.state_requires and {} == f.state_writes for f in g.load.forms()))
        self.assertEqual({}, g.isa_setup().state(adt.FP64))


if __name__ == "__main__":
    unittest.main()
