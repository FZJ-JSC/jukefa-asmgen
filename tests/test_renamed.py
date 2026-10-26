"""
fadd and fmul are add and mul (0.6, phase 1); the old names - the
generator's attributes, the classes' - were deprecated for a release and
are gone (phase 9). opd3 is optransform and move's base class opmove
(phase 10): their old names deprecated for a release.
"""
import warnings
import importlib
import unittest

from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.avx import fma256


class test_renamed(unittest.TestCase):
    def test_generator_attributes_gone(self):
        for gen in (neon, sve, rvv, fma256):
            g = gen()
            with self.subTest(gen=gen.__name__):
                self.assertFalse(hasattr(g, "fadd"))
                self.assertFalse(hasattr(g, "fmul"))
                self.assertTrue(callable(g.add) and callable(g.mul))

    def test_class_names_gone(self):
        for isa in ("neon", "sve", "avx", "rvv"):
            module = importlib.import_module(f"asmgen.asmblocks.{isa}.opd3")
            with self.subTest(isa=isa):
                self.assertTrue(hasattr(module, f"{isa}_add") and hasattr(module, f"{isa}_mul"))
                for old in (f"{isa}_fadd", f"{isa}_fmul"):
                    with self.assertRaises(AttributeError):
                        getattr(module, old)



class test_phase10_names(unittest.TestCase):
    def old(self, module, name):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = getattr(importlib.import_module(module), name)
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught), (module, name))
        return value

    def test_modules(self):
        from asmgen.asmblocks.op.optransform import optransform, optransform_modifier
        from asmgen.asmblocks.op.opmove import opmove, opmove_modifier
        self.assertIs(optransform, self.old("asmgen.asmblocks.op.opd3", "opd3"))
        self.assertIs(optransform_modifier, self.old("asmgen.asmblocks.op.opd3", "opd3_modifier"))
        self.assertIs(opmove, self.old("asmgen.asmblocks.op.move", "move"))
        self.assertIs(opmove_modifier, self.old("asmgen.asmblocks.op.move", "move_modifier"))

    def test_package(self):
        from asmgen.asmblocks import op
        self.assertIs(op.optransform_modifier, self.old("asmgen.asmblocks.op", "opd3_modifier"))
        self.assertIs(op.opmove_modifier, self.old("asmgen.asmblocks.op", "move_modifier"))

    def test_new_names_quiet(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            from asmgen.asmblocks.op import optransform, optransform_modifier, opmove, opmove_modifier  # noqa
            self.assertTrue(issubclass(type(neon().add), optransform))

if __name__ == "__main__":
    unittest.main()
