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
