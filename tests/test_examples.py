"""
The examples keep working: examples/axpy.py, for every ISA it offers, both
arithmetic forms, emits what it emitted when recorded (tests/golden), and
its output assembles where it did
"""
import unittest

from tests.golden.snapshot import ASSEMBLERS, assemble, compare, example_isa, read


class test_examples(unittest.TestCase):
    def test_axpy_unchanged(self):
        changed, gone, added = compare("examples")
        self.assertFalse(changed or gone, f"axpy: {len(changed)} changed, {len(gone)} gone: "
                         + ", ".join((changed + gone)[:6]))

    def test_axpy_assembles(self):
        entries = read("examples")
        for isa in sorted({example_isa(k) for k in entries}):
            if isa not in ASSEMBLERS:
                continue
            mine = {k: v for k, v in entries.items() if example_isa(k) == isa}
            for key, text in mine.items():
                with self.subTest(key=key):
                    if text.startswith("!"):
                        self.skipTest(f"{key}: fails, as recorded")
                    failing = assemble(isa, {key: text})
                    if failing is None:
                        self.skipTest(f"{isa}: no assembler")
                    self.assertFalse(failing, f"{key} doesn't assemble")


if __name__ == "__main__":
    unittest.main()
