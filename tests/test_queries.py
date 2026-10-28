"""
Questions instead of trials (0.6, phase 7): op.fits(**the call's own
arguments) is the signature a call would take, None where it would raise -
nothing raised, no text kept; op.forms(dts=, rtypes=, modifiers=,
operand_modifiers=) the signatures there are for given operands.
"""
import unittest

from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.avx import fma256
from asmgen.asmblocks.op import register_type as rt
from asmgen.asmblocks.op.opmem import opmem_modifier as M
from asmgen.asmblocks.op.opset import opset_modifier as S
from asmgen.registers import asm_data_type as adt


class test_fits(unittest.TestCase):
    def test_the_signature_or_none(self):
        g = rvv()
        sig = g.load.fits(dregs=[g.vreg(0)], areg=g.greg(5), dt=adt.FP64)
        self.assertIsNotNone(sig)
        self.assertEqual(set(), set(sig.modifiers))
        self.assertIsNone(g.load.fits(dregs=[g.vreg(0)], areg=g.greg(5), dt=adt.FP64,
                                      modifiers={M.IOFFSET}, ioffset=8))

    def test_predicates(self):
        g = sve()
        self.assertIsNone(g.load.fits(dregs=[g.vreg(0)], areg=g.greg(0), dt=adt.FP64))
        self.assertIsNotNone(g.load.fits(dregs=[g.vreg(0)], areg=g.greg(0), dt=adt.FP64,
                                         modifiers={M.MASK}, amreg=g.mreg(0)))

    def test_constraints(self):
        g = fma256()
        kw = dict(dregs=[g.greg(9)], areg=g.greg(8), dt=adt.UINT64, modifiers={M.IOFFSET})
        self.assertIsNotNone(g.load.fits(**kw, ioffset=127))
        self.assertIsNone(g.load.fits(**kw, ioffset=128))       # disp8 by default

    def test_argument_errors(self):
        g = neon()
        self.assertIsNone(g.set.fits(dreg=g.greg(1), modifiers={S.CONST}, value=0))   # set needs dt
        self.assertIsNotNone(g.set.fits(dreg=g.greg(1), dt=adt.UINT64, modifiers={S.CONST}, value=0))

    def test_calls_still_raise(self):
        g = rvv()
        with self.assertRaises(ValueError):
            g.load(dregs=[g.vreg(0)], areg=g.greg(5), dt=adt.FP64, modifiers={M.IOFFSET}, ioffset=8)


class test_refusal(unittest.TestCase):
    def test_why_not(self):
        g = rvv()
        kw = dict(dregs=[g.vreg(0)], areg=g.greg(5), dt=adt.FP64)
        self.assertIsNone(g.load.refusal(**kw))
        why = g.load.refusal(**kw, modifiers={M.IOFFSET}, ioffset=8)
        self.assertTrue(why.startswith("ValueError: "), why)      # the ISA's own diagnosis

    def test_agrees_with_fits(self):
        g = fma256()
        for offset in (0, 127, 128, -129):
            kw = dict(dregs=[g.greg(9)], areg=g.greg(8), dt=adt.UINT64, modifiers={M.IOFFSET}, ioffset=offset)
            self.assertEqual(g.load.fits(**kw) is None, g.load.refusal(**kw) is not None, offset)


class test_forms(unittest.TestCase):
    def test_scalar_operand(self):
        forms = rvv().fma.forms(dts={"adreg": adt.FP64}, rtypes={"bdreg": rt.FP})
        self.assertTrue(forms)
        for s in forms:
            self.assertEqual(rt.FP, s.operands["bdreg"].rtype)
            self.assertEqual(adt.FP64, s.operands["adreg"].dt)

    def test_whether_loads_take_a_predicate(self):
        vector_loads = sve().load.forms(rtypes={"adreg": rt.VEC})
        self.assertTrue(vector_loads)
        self.assertTrue(all(M.MASK in s.modifiers for s in vector_loads))
        # RVV's may: masked (v0.t) and not
        rvv_loads = rvv().load.forms(rtypes={"adreg": rt.VEC})
        self.assertTrue(any(M.MASK in s.modifiers for s in rvv_loads))
        self.assertTrue(any(M.MASK not in s.modifiers for s in rvv_loads))

    def test_modifiers(self):
        g = neon()
        offsets = g.load.forms(modifiers={M.IOFFSET})
        self.assertTrue(offsets)
        self.assertTrue(all(M.IOFFSET in s.modifiers for s in offsets))
        self.assertEqual([], g.load.forms(dts={"nosuchoperand": adt.FP64}))

    def test_gated_forms_hidden(self):
        g = fma256()
        full = [s for s in g.load.forms(modifiers={M.IOFFSET}) if "enable-disp32" in s.requires]
        self.assertEqual([], full)                              # disp32 off by default
        g.set_parameter("enable-disp32", "on")
        self.assertTrue([s for s in g.load.forms(modifiers={M.IOFFSET}) if "enable-disp32" in s.requires])


if __name__ == "__main__":
    unittest.main()
