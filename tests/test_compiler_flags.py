"""
Compiler flags from the generator (0.6, phase 7): gen.compiler_flags(compiler)
is its ISA and the extensions it may use (enabled_features) - set_features
narrows them; rvv071's follow XTheadVector. They compile its output.
"""
import os
import shutil
import subprocess
import tempfile
import unittest

from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.rvv071 import rvv071
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.sme import sme
from asmgen.asmblocks.avx import fma256, avx512
from asmgen.asmblocks.noarch import compiler_family
from asmgen.callconv.fngen import fngen
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.op.opset import opset_modifier


def baseline(gen):
    g = gen()
    g.set_features([])
    return g


class test_flags(unittest.TestCase):
    def test_baselines(self):
        for gen, flags in ((rvv, ["-march=rv64gcv"]), (rvv071, ["-march=rv64gc_xtheadvector"]),
                           (neon, ["-march=armv8.2-a"]), (sve, ["-march=armv8.2-a+sve"]),
                           (sme, ["-march=armv9-a+sme"]), (fma256, ["-mavx2", "-mfma"]),
                           (avx512, ["-mavx512f"])):
            with self.subTest(gen=gen.__name__):
                self.assertEqual(flags, baseline(gen).compiler_flags())

    def test_features(self):
        self.assertEqual(["-march=rv64gcv_zba_zfh_zvfh"], rvv().compiler_flags())
        g = sve()
        g.set_features(["SVE2", "I8MM"])
        self.assertEqual(["-march=armv8.2-a+sve+sve2+i8mm"], g.compiler_flags())
        g = avx512()
        g.set_features(["VL", "BW"])
        self.assertEqual(["-mavx512f", "-mavx512vl", "-mavx512bw"], g.compiler_flags())
        self.assertEqual({"VL", "BW"}, g.enabled_features())
        self.assertEqual(set(avx512.FEATURES), avx512().enabled_features())   # no set_features

    def test_xtheadvector(self):
        g = rvv071()
        g.set_parameter("XTheadVector", "off")
        self.assertEqual(["-mepi"], g.compiler_flags())        # EPI's toolchain

    def test_compilers(self):
        for name, family in (("gcc", "gnu"), ("riscv64-linux-gnu-g++", "gnu"), ("/usr/bin/clang-18", "llvm"),
                             ("gcc-13", "gnu"), ("armclang", "arm")):
            self.assertEqual(family, compiler_family(name))
        with self.assertRaises(ValueError):
            compiler_family("icx-nonsense")
        with self.assertRaises(NotImplementedError):
            rvv().compiler_flags("armclang")
        self.assertEqual(["-march=armv8.2-a+sve"], baseline(sve).compiler_flags("armclang"))


# generator, the extensions to enable (ones GCC 13 knows), cross compiler
COMPILES = [
    (rvv, ["ZBA", "ZVFH"], "riscv64-linux-gnu-gcc"),
    (rvv071, ["ZBA"], "riscv64-linux-gnu-gcc"),
    (neon, ["FP16", "FHM", "BF16", "DOTPROD", "I8MM"], "aarch64-linux-gnu-gcc"),
    (sve, ["SVE2", "BF16", "I8MM"], "aarch64-linux-gnu-gcc"),
    (fma256, [], "gcc"),
    (avx512, ["VL", "DQ", "BW", "FP16", "BF16"], "gcc"),
]


def kernel(g):
    """a framed function: a vector register zeroed - an instruction of the
    ISA's vectors (SVE's dup needs +sve), unpredicated everywhere - then the
    return"""
    g.set_output_inline(False)
    body = g.set(dreg=g.vreg(0), dt=adt.FP64, modifiers={opset_modifier.CONST}, constant="zero")
    return fngen(g).function("k", body + g.ret()) + g.asmfile_annotations()


class test_flags_compile(unittest.TestCase):
    def compile(self, cc, flags, text):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "k.s")
            with open(src, "w") as f:
                f.write(text)
            return subprocess.run([cc, *flags, "-c", src, "-o", os.path.join(d, "k.o")],
                                  capture_output=True, text=True)

    def test_baseline_and_features(self):
        for gen, features, cc in COMPILES:
            with self.subTest(gen=gen.__name__):
                if not shutil.which(cc):
                    self.skipTest(f"{cc} not found")
                for chosen in ([], features):
                    g = gen()
                    g.set_features(chosen)
                    r = self.compile(cc, g.compiler_flags(cc), kernel(g))
                    self.assertEqual(0, r.returncode, f"{chosen}: {r.stderr}")

    def test_needed(self):
        cc = "aarch64-linux-gnu-gcc"
        if not shutil.which(cc):
            self.skipTest(f"{cc} not found")
        g = baseline(sve)
        self.assertNotEqual(0, self.compile(cc, ["-march=armv8-a"], kernel(g)).returncode)


if __name__ == "__main__":
    unittest.main()
