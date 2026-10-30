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


class test_vector_length_helper(unittest.TestCase):
    """
    SME's get_simd_size(): the streaming vector length read without a
    switch (rdsvl, encoded) - a caller in streaming mode stays there
    """

    def test_no_switch(self):
        text = sme().c_simd_size_function
        self.assertIn(".inst 0x04bf5820", text)
        self.assertNotIn("smstart", text)
        self.assertNotIn("smstop", text)

    def test_runs(self):
        import os
        import shutil
        import subprocess
        import tempfile
        if not (shutil.which("aarch64-linux-gnu-gcc") and shutil.which("qemu-aarch64-static")):
            self.skipTest("needs aarch64-linux-gnu-gcc and qemu-aarch64-static")
        src = "#include <stddef.h>\n" + sme().c_simd_size_function + \
              "\nint main(void) { return (int) get_simd_size(); }\n"
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "vl.c"), "w") as f:
                f.write(src)
            # C compiled without SME
            r = subprocess.run(["aarch64-linux-gnu-gcc", "-O1", "-static", os.path.join(d, "vl.c"),
                                "-o", os.path.join(d, "vl")], capture_output=True, text=True)
            self.assertEqual(0, r.returncode, r.stderr)
            for bits in (256, 512):
                r = subprocess.run(["qemu-aarch64-static", "-cpu", f"max,sme{bits}=on", os.path.join(d, "vl")])
                self.assertEqual(bits // 8, r.returncode)
