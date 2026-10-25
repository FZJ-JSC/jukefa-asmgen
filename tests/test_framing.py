"""
Framing (0.6, phase 6): fngen.function writes a function - section,
alignment, .global, .type, label, body, .size, data - gen.align aligns what
follows, offset past the boundary by nops, gen.nop is one nop, and
gen.asmfile_annotations is what a file says once (the GNU-stack note).
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest

from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.avx import fma256
from asmgen.callconv.fngen import fngen

# generator, assembler and its flags
ISAS = [
    # no linker relaxation: the object shows the alignment's padding as the
    # linked program will (relaxing, the assembler pads the most it might
    # need and leaves the linker to delete the rest)
    (rvv, ["riscv64-linux-gnu-as", "-march=rv64gcv", "-mno-relax"]),
    (neon, ["aarch64-linux-gnu-as"]),
    (fma256, ["as"]),
]


def standalone(gen):
    g = gen()
    g.set_output_inline(False)
    return g


def symbols(tool, text):
    """name: (value, size, type, bind) of the assembled text's symbols"""
    with tempfile.TemporaryDirectory() as d:
        src, obj = os.path.join(d, "a.s"), os.path.join(d, "a.o")
        with open(src, "w") as f:
            f.write(text)
        r = subprocess.run([*tool, src, "-o", obj], capture_output=True, text=True)
        if r.returncode:
            raise AssertionError(r.stderr + text)
        out = subprocess.run(["readelf", "-sW", obj], capture_output=True, text=True).stdout
    found = {}
    for line in out.splitlines():
        cols = line.split()
        if len(cols) >= 8 and re.match(r"\d+:", cols[0]):
            found[cols[7]] = (int(cols[1], 16), int(cols[2]), cols[3], cols[4])
    return found


class test_framing(unittest.TestCase):
    def test_function(self):
        for gen, tool in ISAS:
            with self.subTest(gen=gen.__name__):
                g = standalone(gen)
                text = fngen(g).function("kernel", g.nop() + g.ret(), align=64) + g.asmfile_annotations()
                lines = text.splitlines()
                self.assertEqual([".text", ".balign 64", ".global kernel", ".type kernel, %function", "kernel:"],
                                 lines[:5])
                self.assertEqual([".size kernel, .-kernel", '.section .note.GNU-stack,"",%progbits'], lines[-2:])
                if not shutil.which(tool[0]) or not shutil.which("readelf"):
                    continue
                value, size, kind, bind = symbols(tool, text)["kernel"]
                self.assertEqual(("FUNC", "GLOBAL"), (kind, bind))
                self.assertGreater(size, 0)

    def test_offset(self):
        for gen, tool in ISAS:
            with self.subTest(gen=gen.__name__):
                g = standalone(gen)
                text = fngen(g).function("kernel", g.ret(), align=64, offset=8)
                if not shutil.which(tool[0]) or not shutil.which("readelf"):
                    continue
                self.assertEqual(8, symbols(tool, text)["kernel"][0] % 64)

    def test_data_after_size(self):
        g = standalone(neon)
        text = fngen(g).function("k", g.ret(), data=g.asmwrap(".section .rodata") + g.asmwrap(".quad 1"))
        self.assertLess(text.index(".size k"), text.index(".section .rodata"))


class test_align(unittest.TestCase):
    def test_offset_by_nops(self):
        for gen, tool in ISAS:
            with self.subTest(gen=gen.__name__):
                g = standalone(gen)
                text = g.asmwrap(".text") + g.nop() + g.align(64, 8) + g.asmwrap("here:") + g.ret()
                if not shutil.which(tool[0]) or not shutil.which("readelf"):
                    continue
                self.assertEqual(64 + 8, symbols(tool, text)["here"][0])

    def test_nop_encodings(self):
        self.assertIn(".fill 2, 4, 0x13", standalone(rvv).align(64, 8))
        self.assertIn(".fill 2, 4, 0xd503201f", standalone(neon).align(64, 8))
        self.assertIn(".fill 8, 1, 0x90", standalone(fma256).align(64, 8))
        self.assertEqual(".balign 64\n", standalone(neon).align(64))

    def test_compressed_nop(self):
        g = standalone(rvv)
        self.assertEqual(".balign 64\n.fill 1, 4, 0x13\n.fill 1, 2, 0x1\n", g.align(64, 6))
        tool = ISAS[0][1]
        if shutil.which(tool[0]) and shutil.which("readelf"):
            text = g.asmwrap(".text") + g.align(64, 6) + g.asmwrap("here:") + g.ret()
            self.assertEqual(6, symbols(tool, text)["here"][0] % 64)

    def test_refused(self):
        with self.assertRaises(ValueError):
            standalone(neon).align(48)            # not a power of 2
        with self.assertRaises(ValueError):
            standalone(neon).align(64, 64)        # past the alignment
        with self.assertRaises(ValueError):
            standalone(neon).align(64, 2)         # AArch64's nops are 4 bytes


class test_inline(unittest.TestCase):
    def test_nothing_said_for_the_file(self):
        g = neon()
        g.set_output_inline(True)
        self.assertEqual("", g.asmfile_annotations())
        self.assertEqual('"nop\\n\\t"\n', g.nop())


if __name__ == "__main__":
    unittest.main()
