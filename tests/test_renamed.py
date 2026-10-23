"""
fadd and fmul became add and mul in 0.6 (docs/V06_REWORK.md, phase 1): the
old names still work for one release - the generators' attributes and the
operations' classes - with a DeprecationWarning
"""
import importlib
import unittest
import warnings

from tests.golden.snapshot import backends


class test_renamed(unittest.TestCase):
    def test_generator_attributes(self):
        for name, ctor in backends().items():
            gen = ctor()
            for old, new in (("fadd", "add"), ("fmul", "mul")):
                with self.subTest(backend=name, name=old):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        self.assertIs(getattr(gen, new), getattr(gen, old))
                    self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

    def test_class_names(self):
        for isa in ("avx", "neon", "rvv", "sve"):
            module = importlib.import_module(f"asmgen.asmblocks.{isa}.opd3")
            for old, new in (("fadd", "add"), ("fmul", "mul")):
                with self.subTest(isa=isa, name=old):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        self.assertIs(getattr(module, f"{isa}_{new}"), getattr(module, f"{isa}_{old}"))
                    self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from asmgen.asmblocks.rvv.opd3 import rvv_fmul  # pylint: disable=import-outside-toplevel
        self.assertEqual("rvv_mul", rvv_fmul.__name__)

    def test_unknown_names_still_fail(self):
        module = importlib.import_module("asmgen.asmblocks.sve.opd3")
        with self.assertRaises(AttributeError):
            getattr(module, "sve_fdiv")


if __name__ == "__main__":
    unittest.main()
