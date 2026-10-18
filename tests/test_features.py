"""
Optional ISA extensions as generator parameters, gating signatures
"""

import unittest

from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.rvv import rvv
from asmgen.registers import asm_data_type as adt


class test_features(unittest.TestCase):

    def test_parameters(self):
        g = neon()
        self.assertIn("FHM", g.get_parameters())
        self.assertEqual("off", g.get_param_value("FHM"))
        g.set_parameter("FHM", "on")
        self.assertEqual("on", g.get_param_value("FHM"))

    def test_ungated_by_default(self):
        # nothing hidden until features are set
        self.assertTrue(any(s.operands["adreg"].dt == adt.FP16 for s in neon().fma.get_signatures()))

    def test_baseline(self):
        g = neon()
        g.set_features([])
        self.assertFalse(any(adt.FP16 in {o.dt for o in s.operands.values() if hasattr(o, "dt")}
                             for s in g.fma.get_signatures()))
        g.set_parameter("FHM", "on")
        self.assertTrue(any(s.operands["adreg"].dt == adt.FP16 and s.operands["cdreg"].dt == adt.FP32
                            for s in g.fma.get_signatures()))

    def test_sve2_and_zvfh(self):
        g = sve()
        g.set_features([])
        self.assertFalse(any("PART" in {m.name for m in s.modifiers} for s in g.fma.get_signatures()))
        g.set_features(["SVE2"])
        self.assertTrue(any("PART" in {m.name for m in s.modifiers} for s in g.fma.get_signatures()))
        r = rvv()
        r.set_features([])
        self.assertFalse(any(s.operands["adreg"].dt == adt.FP16 for s in r.fma.get_signatures()))

    def test_unknown(self):
        with self.assertRaises(ValueError):
            neon().set_features(["SVE2"])


if __name__ == "__main__":
    unittest.main()


class test_features_per_isa(unittest.TestCase):
    """
    What needs an optional extension requires it; what doesn't exist isn't offered
    """

    def rows(self, op):
        out = {}
        for s in op.get_signatures():
            c = s.operands["cdreg"].dt.name
            out[f"{s.operands['adreg'].dt.name}->{c}"] = set(s.requires)
        return out

    def test_neon(self):
        g = neon()
        fma, fadd = self.rows(g.fma), self.rows(g.fadd)
        self.assertEqual({"FHM"}, fma["HALF->SINGLE"])
        self.assertEqual({"FP8FMA"}, fma["FP8E4M3->HALF"])        # not FHM
        self.assertNotIn("HALF->SINGLE", fadd)                    # no faddl for floats
        self.assertNotIn("FP8E4M3->FP8E4M3", fma)                 # FP8 only widens

    def test_sve(self):
        fma, fmul = self.rows(sve().fma), self.rows(sve().fmul)
        self.assertEqual({"BF16"}, fma["BF16->SINGLE"])           # bfmlalb: no SVE2
        self.assertEqual({"B16B16"}, fmul["BF16->BF16"])
        self.assertEqual(set(), fma["HALF->HALF"])
        self.assertNotIn("HALF->SINGLE", fmul)

    def test_rvv(self):
        fma = self.rows(rvv().fma)
        self.assertFalse([k for k in fma if "FP8" in k])
        self.assertEqual({"ZVFH"}, fma["HALF->SINGLE"])
        self.assertEqual(set(), fma["SINGLE->DOUBLE"])

    def test_avx512(self):
        from asmgen.asmblocks.avx import avx512
        g = avx512()
        self.assertEqual({"FP16"}, self.rows(g.fma)["HALF->HALF"])
        lanes = {s.operands["adreg"].dt.name: set(s.requires) for s in g.load.get_signatures()
                 if any("ILANE" == m.name for m in s.operands["adreg"].modifiers)}
        self.assertEqual({"BW"}, lanes["SINT16"])
        self.assertEqual({"DQ"}, lanes["SINT32"])
        self.assertEqual(set(), lanes["DOUBLE"])

    def test_sme(self):
        from asmgen.asmblocks.sme import sme
        g = sme()
        rows = {f"{s.operands['adreg'].dt.name}->{s.operands['cdreg'].dt.name}": set(s.requires)
                for s in g.fopa.get_signatures()}
        self.assertEqual({"F64F64"}, rows["DOUBLE->DOUBLE"])
        self.assertEqual(set(), rows["HALF->SINGLE"])
        self.assertNotIn("FP8E4M3->FP8E4M3", rows)
