"""
Unwind information (CFI) from fngen: .cfi_startproc and .cfi_endproc around
a function, the stack adjustment and each saved register after the saves
(callconv) - checked as the assembler encodes it (readelf's frames) on
x86, AArch64 and RISC-V. Inline assembly gets none: its CFI is the
compiler's.
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest

from asmgen.asmblocks.avx import fma256
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.rvv import rvv
from asmgen.callconv.fngen import fngen

# generator, compiler, callee-saved general registers to clobber (indices)
CASES = (
    (fma256, "gcc", 3),
    (neon, "aarch64-linux-gnu-gcc", 3),
    (rvv, "riscv64-linux-gnu-gcc", 3),
)


def function(gen, saved, cfi=None, inline=False):
    """a function clobbering saved callee-saved registers: its text, and
    the registers' names"""
    g = gen()
    g.set_output_inline(inline)
    rt = g.get_reg_tracker()
    fn = fngen(g, rt, cfi=cfi)
    cc = g.create_callconv()
    fn.init_cc(cc)
    sp = cc.spreg
    regs = [i for i in cc.callee_save_lists["greg"]
            if i != sp and i not in rt.used_regs["greg"]][:saved]     # free ones
    for i in regs:
        rt.reserve_specific_reg("greg", i)
    save, load, restore = fn.get_boilerplate(cc=cc)
    text = fn.function("k", save + load + restore + g.ret()) + g.asmfile_annotations()
    return text, [str(g.greg(i)) for i in regs]


def frames(cc, text):
    with tempfile.TemporaryDirectory() as d:
        src, obj = os.path.join(d, "k.s"), os.path.join(d, "k.o")
        with open(src, "w") as f:
            f.write(text)
        subprocess.run([cc, "-c", src, "-o", obj], check=True, capture_output=True)
        return subprocess.run(["readelf", "--debug-dump=frames", obj], check=True,
                              capture_output=True, text=True).stdout


class test_cfi(unittest.TestCase):
    def test_unwind_tables(self):
        if not shutil.which("readelf"):
            self.skipTest("needs readelf")
        for gen, cc, saved in CASES:
            with self.subTest(gen=gen.__name__):
                if not shutil.which(cc):
                    self.skipTest(f"needs {cc}")
                text, names = function(gen, saved)
                self.assertIn(".cfi_startproc", text)
                dump = frames(cc, text)
                # the adjustment: the frame grows, then shrinks back
                offsets = [int(o) for o in re.findall(r"DW_CFA_def_cfa_offset: (\d+)", dump)]
                self.assertGreaterEqual(len(offsets), 2, dump)
                self.assertGreater(max(offsets), min(offsets))
                # each saved register is somewhere in the frame, then restored
                for name in names:
                    self.assertRegex(dump, rf"DW_CFA_offset: \S+ \({name}\) at cfa-\d+")
                    self.assertRegex(dump, rf"DW_CFA_restore: \S+ \({name}\)")

    def test_none_inline(self):
        text, _ = function(neon, 2, inline=True)
        self.assertNotIn(".cfi", text)

    def test_turned_off(self):
        text, _ = function(neon, 2, cfi=False)
        self.assertNotIn(".cfi", text)

    def test_no_saves(self):
        g = neon()
        g.set_output_inline(False)
        text = fngen(g).function("k", g.ret())
        self.assertEqual(1, text.count(".cfi_startproc"))
        self.assertEqual(1, text.count(".cfi_endproc"))
        self.assertNotIn(".cfi_adjust", text)


if __name__ == "__main__":
    unittest.main()
