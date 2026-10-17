"""
Moves between registers: NEON and RVV forms, SVE's lane range
"""

import os
import shutil
import subprocess
import tempfile
import unittest

from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.op import operand_modifier as omod
from asmgen.registers import asm_data_type as adt

B, L = omod.BCAST, omod.ILANE


def plain(text):
    return text.strip().strip('"').replace("\\n", "").replace("\\t", "").strip()


def forms(g, dt, scalar):
    v = g.vreg
    return {
        "copy": dict(dregs=[v(1), v(2)], dts=[dt] * 2),
        "scalar to all": dict(dregs=[scalar, v(2)], dts=[dt] * 2, operand_modifiers={"bdreg": {B}}),
        "lane to all": dict(dregs=[v(1), v(2)], dts=[dt] * 2,
                            operand_modifiers={"adreg": {L}, "bdreg": {B}}, adreg_lane=1),
        "scalar to lane": dict(dregs=[scalar, v(2)], dts=[dt] * 2,
                               operand_modifiers={"bdreg": {L}}, bdreg_lane=0),
    }


class test_neon_moves(unittest.TestCase):

    def test_forms(self):
        g = neon()
        out = {k: plain(g.move(**kw)) for k, kw in forms(g, adt.FP64, g.freg(3, adt.FP64)).items()}
        self.assertEqual("mov v2.16b,v1.16b", out["copy"])
        self.assertEqual("dup v2.2d,v3.d[0]", out["scalar to all"])
        self.assertEqual("dup v2.2d,v1.d[1]", out["lane to all"])
        self.assertEqual("ins v2.d[0],v3.d[0]", out["scalar to lane"])

    def test_lane_range(self):
        g = neon()
        with self.assertRaises(Exception):
            g.move(dregs=[g.vreg(1), g.vreg(2)], dts=[adt.FP64] * 2,
                   operand_modifiers={"adreg": {L}, "bdreg": {B}}, adreg_lane=2)

    @unittest.skipUnless(shutil.which("aarch64-linux-gnu-as"), "needs an AArch64 assembler")
    def test_assembles(self):
        g = neon()
        lines = []
        for dt, scalar in ((adt.FP64, g.freg(3, adt.FP64)), (adt.FP32, g.freg(3, adt.FP32)),
                           (adt.SINT32, g.greg(4))):
            lines += [plain(g.move(**kw)) for kw in forms(g, dt, scalar).values()]
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "m.s")
            with open(src, "w") as f:
                f.write("\n".join(lines) + "\n")
            r = subprocess.run(["aarch64-linux-gnu-as", src, "-o", os.path.join(d, "m.o")],
                               capture_output=True, text=True)
            self.assertEqual(0, r.returncode, r.stderr)


class test_rvv_moves(unittest.TestCase):

    def test_forms(self):
        g = rvv()
        out = {k: plain(g.move(**kw)) for k, kw in forms(g, adt.FP64, g.freg(3, adt.FP64)).items()}
        self.assertEqual("vmv.v.v v2,v1", out["copy"])
        self.assertEqual("vfmv.v.f v2,f3", out["scalar to all"])
        self.assertEqual("vrgather.vi v2,v1,1", out["lane to all"])
        self.assertEqual("vfmv.s.f v2,f3", out["scalar to lane"])

    def test_gather_needs_distinct_registers(self):
        g = rvv()
        with self.assertRaises(Exception):
            g.move(dregs=[g.vreg(1), g.vreg(1)], dts=[adt.FP64] * 2,
                   operand_modifiers={"adreg": {L}, "bdreg": {B}}, adreg_lane=1)


class test_sve_lane_range(unittest.TestCase):

    def test_dup_indexed(self):
        # dup z.d, z.d[i]: i within 0..7 (512 bits of doubles)
        sig = next(s for s in sve().move.get_signatures()
                   if not s.modifiers and "adreg_lane" in s.operands
                   and s.operands["adreg"].dt == adt.FP64)
        check = lambda v: [c("adreg_lane", set(), v, {})
                           for c in sig.operands["adreg_lane"].value_constraints]
        check(7)
        with self.assertRaises(ValueError):
            check(8)


if __name__ == "__main__":
    unittest.main()
