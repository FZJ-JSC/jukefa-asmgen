"""
fadd and fmul are add and mul (0.6, phase 1); the old names - the
generator's attributes, the classes' - were deprecated for a release and
are gone (phase 9).
"""
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


if __name__ == "__main__":
    unittest.main()
