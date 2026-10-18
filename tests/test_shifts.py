"""
Moves shifting and rotating lanes and blocks: every form of every ISA
assembles; irot negative moves towards lane 0
"""

import os
import shutil
import subprocess
import tempfile
import unittest

from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.avx import avx512
from asmgen.registers import asm_data_type as adt

SHIFTS = {"ILANEROT", "ILANESHIFT", "IBLOCKROT"}


def forms(g):
    out = []
    for s in g.move.get_signatures():
        mods = s.operands["adreg"].modifiers
        if not SHIFTS & {m.name for m in mods} or s.operands["adreg"].dt not in (adt.FP64, adt.FP32):
            continue
        c = s.operands["adreg_irot"].value_constraints[0]
        dt = s.operands["adreg"].dt
        for irot in sorted({c.minval, c.maxval}):
            kw = dict(dregs=[g.vreg(3), g.vreg(3 if s.operands["bdreg"].value_constraints else 5)],
                      dts=[dt, dt], modifiers=set(), operand_modifiers={"adreg": set(mods)},
                      adreg_irot=irot, **s.structural_params)
            if "fillreg" in s.operands:
                kw["fillreg"] = g.freg(1, dt)
            out.append(g.move(**kw).strip())
    return out


class test_shifts(unittest.TestCase):

    def assembles(self, lines, asm):
        if not shutil.which(asm[0]):
            self.skipTest(f"needs {asm[0]}")
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "k.s")
            with open(src, "w") as f:
                f.write("\n".join(lines) + "\n")
            r = subprocess.run(asm + [src, "-o", os.path.join(d, "k.o")], capture_output=True, text=True)
        self.assertEqual(0, r.returncode, r.stderr)

    def gen(self, cls):
        g = cls()
        g.output_inline = False
        return g

    def test_rvv(self):
        lines = forms(self.gen(rvv))
        self.assertIn("vslidedown.vi v5,v3,1", lines)
        self.assertTrue(any(l.startswith("vfslide1down.vf") for l in lines))
        self.assertTrue(lines)
        self.assertTrue(all(not l.startswith("vslideup") for l in lines))
        self.assertTrue(lines) and self.assembles(lines, ["riscv64-linux-gnu-as", "-march=rv64gcv"])

    def test_neon(self):
        lines = forms(self.gen(neon))
        self.assertIn("ext v5.16b,v3.16b,v3.16b,#8", lines)   # irot -1 on doubles
        self.assembles(lines, ["aarch64-linux-gnu-as"])

    def test_sve(self):
        lines = forms(self.gen(sve))
        self.assertIn("ext z3.b,z3.b,z3.b,#8", lines)         # destructive
        self.assertTrue(any(l.startswith("insr") for l in lines))
        self.assembles(lines, ["aarch64-linux-gnu-as", "-march=armv8-a+sve"])

    def test_avx512(self):
        lines = forms(self.gen(avx512))
        self.assertIn("valignq $1,%zmm3,%zmm3,%zmm5", lines)
        self.assertTrue(any(l.startswith("vshuff64x2") for l in lines))
        self.assembles(lines, ["as"])


if __name__ == "__main__":
    unittest.main()


class test_fill_drain(unittest.TestCase):
    """
    A fill loop's step - a scalar into a vector shifted in place - and a drain
    loop's - lane 0 into a scalar
    """

    def test_rvv(self):
        from asmgen.asmblocks.op import operand_modifier as om
        g = rvv()
        g.output_inline = False
        dt = adt.FP64
        fill = g.move(dregs=[g.freg(1, dt), g.vreg(4)], dts=[dt, dt], modifiers=set(),
                      operand_modifiers={"bdreg": {om.ILANESHIFT, om.FILLOTHERS}}, bdreg_irot=-1)
        drain = g.move(dregs=[g.vreg(4), g.freg(2, dt)], dts=[dt, dt], modifiers=set(),
                       operand_modifiers={"adreg": {om.ILANE}}, adreg_lane=0)
        self.assertEqual("vfslide1down.vf v4,v4,f1", fill.strip())
        self.assertEqual("vfmv.f.s f2,v4", drain.strip())

    def test_sve(self):
        from asmgen.asmblocks.op import operand_modifier as om
        g = sve()
        g.output_inline = False
        dt = adt.FP64
        fill = g.move(dregs=[g.freg(1, dt), g.vreg(4)], dts=[dt, dt], modifiers=set(),
                      operand_modifiers={"bdreg": {om.ILANESHIFT, om.FILLOTHERS}}, bdreg_irot=1)
        self.assertEqual("insr z4.d,d1", fill.strip())
