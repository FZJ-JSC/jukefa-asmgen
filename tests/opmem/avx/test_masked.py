"""
AVX-512's plain loads and stores under a k mask (MASK): a last vector's
first lanes read (the others zeroed) or written - the mask of a run-time
count from MASKAVL (movq, bzhiq, kmovw)
"""
import os
import subprocess
import tempfile
import unittest

from asmgen.asmblocks.avx import avx512
from asmgen.asmblocks.op.opmem import opmem_modifier as mod
from asmgen.asmblocks.op.opset import opset_modifier
from asmgen.registers import asm_data_type as adt


def gen():
    g = avx512()
    g.set_output_inline(False)
    return g


class test_masked(unittest.TestCase):
    def test_forms(self):
        g = gen()
        z, k1, rdi = g.vreg(3), g.mreg(1), g.greg(13)
        call = lambda action, dt, **kw: getattr(g, action)(dregs=[z], areg=rdi, dt=dt, amreg=k1, **kw).strip()
        self.assertEqual("vmovupd (%rdi), %zmm3{%k1}{z}", call("load", adt.FP64, modifiers={mod.MASK}))
        self.assertEqual("vmovupd %zmm3, (%rdi){%k1}", call("store", adt.FP64, modifiers={mod.MASK}))
        # a mask is a lane's: the move of the elements' size
        self.assertEqual("vmovdqu32 (%rdi), %zmm3{%k1}{z}", call("load", adt.SINT32, modifiers={mod.MASK}))
        self.assertEqual("vmovupd 64(%rdi), %zmm3{%k1}{z}",
                         call("load", adt.FP64, modifiers={mod.MASK, mod.IOFFSET}, ioffset=64))
        # with an index register (x86's base + index x scale): kernels'
        # columns, lda apart
        self.assertEqual("vmovupd (%rdi,%r8,1), %zmm3{%k1}{z}",
                         call("load", adt.FP64, modifiers={mod.MASK, mod.GOFFSET, mod.SCALEOFFSET},
                              offreg=g.greg(0), scale_factor=1))
        self.assertEqual("vmovupd %zmm3, 64(%rdi,%r8,8){%k1}",
                         call("store", adt.FP64, modifiers={mod.MASK, mod.GOFFSET, mod.SCALEOFFSET, mod.IOFFSET},
                              offreg=g.greg(0), scale_factor=8, ioffset=64))
        self.assertIn("other than k0", g.load.refusal(dregs=[z], areg=rdi, dt=adt.FP64, modifiers={mod.MASK},
                                                      amreg=g.mreg(0)))
        words = [s for s in g.load.get_signatures() if mod.MASK in s.modifiers and mod.VINDEX not in s.modifiers
                 and adt.SINT16 == s.operands["adreg"].dt]
        self.assertTrue(words and all("BW" in s.requires for s in words))

    def test_semantics(self):
        # run here: every count 0 to 8 - its mask set, a vector loaded and
        # stored under it: the first count doubles copied, the rest left
        flags = open("/proc/cpuinfo").read() if os.path.exists("/proc/cpuinfo") else ""
        if not all(f" {f}" in flags for f in ("avx512f", "bmi2")):
            self.skipTest("needs AVX512F and BMI2")
        g = gen()
        k1, z0 = g.mreg(1), g.vreg(0)
        rdi, rsi, rdx, rax = g.greg(13), g.greg(12), g.greg(11), g.greg(8)
        asm = ("\t.text\n\t.globl f\n\t.type f, @function\nf:\n"
               + g.set(dreg=k1, dt=adt.FP64, modifiers={opset_modifier.MASKAVL}, avlreg=rdx, tmpreg=rax)
               + g.load(dregs=[z0], areg=rdi, dt=adt.FP64, modifiers={mod.MASK}, amreg=k1)
               + g.store(dregs=[z0], areg=rsi, dt=adt.FP64, modifiers={mod.MASK}, amreg=k1)
               + "\tvzeroupper\n\tret\n\t.section .note.GNU-stack,\"\",@progbits\n")
        c = r"""
#include <stdio.h>
void f(const double* src, double* out, long count);
int main(void) {
    int bad = 0;
    for (long count = 0; count <= 8; ++count) {
        double src[8], out[8];
        for (int i = 0; i < 8; ++i) { src[i] = i + 1; out[i] = -1; }
        f(src, out, count);
        for (int i = 0; i < 8; ++i) if (out[i] != (i < count ? i + 1 : -1)) ++bad;
    }
    printf("bad %d\n", bad);
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "f.s"), "w").write(asm)
            open(os.path.join(d, "m.c"), "w").write(c)
            r = subprocess.run(["gcc", "-O1", os.path.join(d, "m.c"), os.path.join(d, "f.s"), "-o",
                                os.path.join(d, "run")], capture_output=True, text=True)
            self.assertEqual(0, r.returncode, r.stderr)
            out = subprocess.run([os.path.join(d, "run")], capture_output=True, text=True, timeout=60).stdout
            self.assertIn("bad 0", out)


if __name__ == "__main__":
    unittest.main()
