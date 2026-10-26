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


    def test_rvv_lmul_runs(self):
        # it broke unnoticed (old register counts, the VF modifier): ran by nothing
        import os, subprocess, sys
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        r = subprocess.run([sys.executable, os.path.join(root, "examples", "rvv_lmul.py")],
                           capture_output=True, text=True, timeout=120,
                           env={**os.environ, "PYTHONPATH": os.path.join(root, "src")})
        self.assertEqual(0, r.returncode, r.stderr[-2000:])
        for lmul in (1, 2, 4, 8):
            self.assertIn(f"e64, m{lmul}", r.stdout)
        self.assertEqual(4, r.stdout.count("vfmacc.vf"))
        self.assertEqual(4, r.stdout.count("vfmacc.vv"))

if __name__ == "__main__":
    unittest.main()
