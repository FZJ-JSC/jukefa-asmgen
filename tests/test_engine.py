"""
SME's engine - streaming mode and ZA - as state: set's ENGINE forms write it
(smstart, smstop), the setup's own switches left out where a caller places
them (engine=False)
"""

import unittest

from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.sme import sme
from asmgen.asmblocks.op.opset import opset_modifier as setmod
from asmgen.registers import asm_data_type as adt


class test_engine(unittest.TestCase):

    def test_forms(self):
        g = sme()
        forms = {f.structural_params["engine"]: f.state_writes for f in g.set.forms()
                 if setmod.ENGINE in f.modifiers}
        self.assertEqual({"on": {"sm": "1", "za": "1"}, "off": {"sm": "0", "za": "0"}}, forms)
        self.assertIn("smstart", g.set(dt=adt.FP64, modifiers={setmod.ENGINE}, engine="on"))
        self.assertIn("smstop", g.set(dt=adt.FP64, modifiers={setmod.ENGINE}, engine="off"))

    def test_setup_without_switches(self):
        g = sme()
        isa = g.isa_setup()
        self.assertEqual(({"sm": "1", "za": "1"}, {"sm": "0", "za": "0"}), isa.engine())
        with_switch = isa.setup(rt=None, dt=adt.FP64)
        without = isa.setup(rt=None, dt=adt.FP64, engine=False)
        self.assertIn("smstart", with_switch)
        self.assertNotIn("smstart", without)
        self.assertEqual(with_switch.replace(g.asmwrap("smstart"), ""), without)
        self.assertIn("smstop", isa.teardown(rt=None, dt=adt.FP64))
        self.assertNotIn("smstop", isa.teardown(rt=None, dt=adt.FP64, engine=False))

    def test_none_elsewhere(self):
        self.assertIsNone(neon().isa_setup().engine())


if __name__ == "__main__":
    unittest.main()
