"""
Immediate constraints on memory operation signatures, against the encodings
"""

import unittest

from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.op import opmem_modifier as mod, operand_modifier as omod
from asmgen.registers import asm_data_type as adt


class test_neon(unittest.TestCase):

    def load(self, **kw):
        g = neon()
        return g.load(dregs=[g.vreg(1)], areg=g.greg(6), dt=adt.FP64, **kw)

    def test_immediate_offset(self):
        # ldr q: -256..255 unscaled (ldur), or multiples of 16 up to 65520
        for ok in (-256, -16, 17, 255, 0, 272, 65520):
            with self.subTest(ok=ok):
                self.assertIn("ldr", self.load(modifiers={mod.IOFFSET}, ioffset=ok))
        for bad in (-272, 273, 65536):
            with self.subTest(bad=bad):
                with self.assertRaises(Exception):
                    self.load(modifiers={mod.IOFFSET}, ioffset=bad)

    def test_vector_offset(self):
        for ok in (-16, 4095):
            self.load(modifiers={mod.VOFFSET}, voffset=ok)
        for bad in (-17, 4096):
            with self.assertRaises(Exception):
                self.load(modifiers={mod.VOFFSET}, voffset=bad)

    def test_post_increment(self):
        # structure loads post-increment by exactly what they transfer
        self.load(modifiers={mod.POSTINC}, iinc=16)
        with self.assertRaises(Exception):
            self.load(modifiers={mod.POSTINC}, iinc=32)
        bcast = {"adreg": {omod.BCAST}}
        self.load(modifiers={mod.POSTINC}, operand_modifiers=bcast, iinc=8)
        with self.assertRaises(Exception):
            self.load(modifiers={mod.POSTINC}, operand_modifiers=bcast, iinc=16)


class test_sve(unittest.TestCase):

    def constraint_check(self, mods, name, value):
        sig = next(s for s in sve().load.get_signatures()
                   if set(s.modifiers) == mods and name in s.operands
                   and any(o.dt == adt.FP64 for o in s.operands.values()))
        try:
            for c in sig.operands[name].value_constraints:
                c(name, set(sig.modifiers), value, {})
            return True
        except ValueError:
            return False

    def test_mul_vl(self):
        mods = {mod.MASK, mod.VOFFSET}
        self.assertTrue(self.constraint_check(mods, "voffset", -8))
        self.assertTrue(self.constraint_check(mods, "voffset", 7))
        self.assertFalse(self.constraint_check(mods, "voffset", 8))

    def test_gather_offset(self):
        mods = {mod.MASK, mod.VINDEX, mod.IOFFSET}
        self.assertTrue(self.constraint_check(mods, "ioffset", 248))
        self.assertFalse(self.constraint_check(mods, "ioffset", 256))
        self.assertFalse(self.constraint_check(mods, "ioffset", 12))


if __name__ == "__main__":
    unittest.main()
