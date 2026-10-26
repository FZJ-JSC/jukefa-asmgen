"""
fadd and fmul are add and mul (0.6, phase 1); the old names - the
generator's attributes, the classes' - were deprecated for a release and
are gone (phase 9). opd3 is optransform and move's base class opmove
(phase 10): their old names gone, the ISAs' packages renamed too.
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
            module = importlib.import_module(f"asmgen.asmblocks.{isa}.optransform")
            with self.subTest(isa=isa):
                self.assertTrue(hasattr(module, f"{isa}_add") and hasattr(module, f"{isa}_mul"))
                for old in (f"{isa}_fadd", f"{isa}_fmul"):
                    with self.assertRaises(AttributeError):
                        getattr(module, old)



class test_phase10_names(unittest.TestCase):
    """
    opd3 is optransform, move's base class opmove (phase 10): the old
    modules, packages and names are gone
    """
    def test_old_modules_gone(self):
        for module in ["asmgen.asmblocks.op.opd3", "asmgen.asmblocks.op.move"] + \
                [f"asmgen.asmblocks.{isa}.{old}" for isa in ("neon", "sve", "sme", "aarch64", "riscv64",
                                                              "x86", "avx", "rvv") for old in ("opd3", "move")]:
            with self.subTest(module=module), self.assertRaises(ModuleNotFoundError):
                importlib.import_module(module)

    def test_old_names_gone(self):
        from asmgen.asmblocks import op
        for name in ("opd3", "opd3_modifier", "move_modifier", "dummy_opd3", "dummy_move"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(op, name))

    def test_new_names(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            from asmgen.asmblocks.op import optransform, optransform_modifier, opmove, opmove_modifier  # noqa
            self.assertTrue(issubclass(type(neon().add), optransform))
            self.assertTrue(issubclass(type(neon().move), opmove))

if __name__ == "__main__":
    unittest.main()
