"""
Two sources permuted (opmove PERMUTE): zip, unzip, transpose units of some
bits, within 128-bit lanes or across the register - the rounds of an
in-register transpose. Every form is run - natively (AVX2, AVX-512) or
under qemu (NEON) - on two vectors of distinct elements and compared with
permuted(), the forms' meaning.
"""
import os
import shutil
import subprocess
import tempfile
import unittest

from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.avx import fma256, avx512
from asmgen.asmblocks.op.opmove import opmove_modifier as MV, permuted, PERMUTE_PATTERNS
from asmgen.registers import asm_data_type as adt, adt_size

HARNESS = r"""
#include <stdio.h>
#include <stdint.h>
typedef uint%(bits)d_t e;
void p(const e* a, const e* b, e* out);
int main(void) {
    e a[%(n)d], b[%(n)d], out[%(n)d];
    for (int i = 0; i < %(n)d; ++i) { a[i] = 1 + i; b[i] = 101 + i; out[i] = 0; }
    p(a, b, out);
    for (int i = 0; i < %(n)d; ++i) printf("%%u ", (unsigned)out[i]);
    printf("\n");
    return 0;
}
"""


def cpu_has(flag):
    try:
        return any(flag in line.split() for line in open("/proc/cpuinfo") if line.startswith("flags"))
    except OSError:
        return False


def function(gen, isa, text, vbytes):
    """p(a, b, out): two vectors loaded, permuted, stored"""
    if "neon" == isa:
        body = ["ldr q0, [x0]", "ldr q1, [x1]", text, "str q2, [x2]", "ret"]
    else:
        r = {32: "ymm", 64: "zmm"}[vbytes]
        body = [f"vmovupd (%rdi), %{r}0", f"vmovupd (%rsi), %{r}1", text,
                f"vmovupd %{r}2, (%rdx)", "vzeroupper", "ret"]
    return ".text\n.global p\n.type p, %function\np:\n" + "\n".join(body) + "\n"


class test_permute(unittest.TestCase):
    def run_forms(self, gen_class, isa, cc, run=()):
        g = gen_class()
        g.set_output_inline(False)
        vbytes = 16 if "neon" == isa else g.simd_size
        checked = 0
        with tempfile.TemporaryDirectory() as d:
            for dt in (adt.FP64, adt.FP32):
                ebits = 8 * adt_size(dt)
                n = vbytes * 8 // ebits
                with open(os.path.join(d, "h.c"), "w") as f:
                    f.write(HARNESS % {"bits": ebits, "n": n})
                for s in g.move.forms(modifiers={MV.PERMUTE}, dts={"adreg": dt}):
                    sp = s.structural_params
                    text = g.move(dregs=[g.vreg(0), g.vreg(1), g.vreg(2)], dts=[dt] * 3,
                                  modifiers={MV.PERMUTE}, **sp).strip()
                    with self.subTest(isa=isa, dt=dt.name, form=text):
                        with open(os.path.join(d, "p.s"), "w") as f:
                            f.write(function(g, isa, text, vbytes))
                        exe = os.path.join(d, "run")
                        r = subprocess.run([*cc, os.path.join(d, "h.c"), os.path.join(d, "p.s"), "-o", exe],
                                           capture_output=True, text=True)
                        self.assertEqual(0, r.returncode, r.stderr)
                        out = subprocess.run([*run, exe], capture_output=True, text=True, timeout=60)
                        got = [int(x) for x in out.stdout.split()]
                        a, b = list(range(1, n + 1)), list(range(101, 101 + n))
                        want = permuted(sp["pattern"], a, b, element_bits=ebits, bits=sp["bits"],
                                        segment=sp["segment"])
                        self.assertEqual(want, got)
                        checked += 1
        return checked

    def test_avx2(self):
        if not (shutil.which("gcc") and cpu_has("avx2")):
            self.skipTest("needs gcc and an AVX2 CPU")
        self.assertGreater(self.run_forms(fma256, "avx2", ["gcc", "-mavx2"]), 0)

    def test_avx512(self):
        if not (shutil.which("gcc") and cpu_has("avx512f")):
            self.skipTest("needs gcc and an AVX-512 CPU")
        self.assertGreater(self.run_forms(avx512, "avx512", ["gcc", "-mavx512f"]), 0)

    def test_neon(self):
        if not (shutil.which("aarch64-linux-gnu-gcc") and shutil.which("qemu-aarch64-static")):
            self.skipTest("needs aarch64-linux-gnu-gcc and qemu-aarch64-static")
        self.assertGreater(self.run_forms(neon, "neon", ["aarch64-linux-gnu-gcc", "-static"],
                                          run=("qemu-aarch64-static", "-cpu", "max")), 0)

    def test_reference(self):
        # the patterns on 4 units, the whole register and 128-bit lanes
        a, b = [0, 1, 2, 3], [10, 11, 12, 13]
        whole = {"zip_lo": [0, 10, 1, 11], "zip_hi": [2, 12, 3, 13], "uzp_even": [0, 2, 10, 12],
                 "uzp_odd": [1, 3, 11, 13], "trn_even": [0, 10, 2, 12], "trn_odd": [1, 11, 3, 13]}
        for pattern in PERMUTE_PATTERNS:
            self.assertEqual(whole[pattern], permuted(pattern, a, b, element_bits=64, bits=64))
        # x86's vunpcklpd: within each 128-bit lane
        self.assertEqual([0, 10, 2, 12], permuted("zip_lo", a, b, element_bits=64, bits=64, segment=128))


if __name__ == "__main__":
    unittest.main()
