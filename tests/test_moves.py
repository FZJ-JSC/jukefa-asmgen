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


from asmgen.asmblocks.avx import avx512, fma256
from asmgen.asmblocks.op import opmem_modifier as mm

BLK = {omod.BLOCK, omod.BCAST}


class test_sve_blocks(unittest.TestCase):

    def ld1rq(self, **kw):
        g = sve()
        return plain(g.load(dregs=[g.vreg(1)], areg=g.greg(2), dt=adt.FP64,
                            operand_modifiers={"adreg": BLK}, amreg=g.mreg(0),
                            adreg_blocksize=2, **kw))

    def test_ld1rq(self):
        self.assertEqual("ld1rqd {z1.d}, p0/z, [x2]", self.ld1rq(modifiers={mm.MASK}))
        self.assertEqual("ld1rqd {z1.d}, p0/z, [x2, #112]",
                         self.ld1rq(modifiers={mm.MASK, mm.IOFFSET}, ioffset=112))
        for bad in (128, 8):
            with self.assertRaises(Exception):
                self.ld1rq(modifiers={mm.MASK, mm.IOFFSET}, ioffset=bad)

    def test_dup_block(self):
        g = sve()
        call = lambda i: plain(g.move(dregs=[g.vreg(3), g.vreg(4)], dts=[adt.FP64] * 2,
                                      operand_modifiers={"adreg": {omod.IBLOCK}, "bdreg": BLK},
                                      adreg_block=i, adreg_blocksize=2, bdreg_blocksize=2))
        self.assertEqual("dup z4.q,z3.q[3]", call(3))
        with self.assertRaises(Exception):
            call(4)


class test_avx_moves(unittest.TestCase):

    def forms(self, g):
        v = g.vreg
        two = [adt.FP64] * 2
        return [plain(g.move(dregs=[v(1), v(2)], dts=two)),
                plain(g.move(dregs=[g.freg(3, adt.FP64), v(2)], dts=two,
                             operand_modifiers={"bdreg": {B}})),
                plain(g.move(dregs=[v(1), v(2)], dts=two,
                             operand_modifiers={"adreg": {omod.IBLOCK}, "bdreg": BLK},
                             adreg_block=1, adreg_blocksize=2, bdreg_blocksize=2)),
                plain(g.move(dregs=[v(1), v(2)], dts=two, operand_modifiers={"adreg": {omod.IBLOCK}},
                             adreg_block=1, adreg_blocksize=2)),
                plain(g.move(dregs=[v(1), v(2)], dts=two, operand_modifiers={"bdreg": {omod.IBLOCK}},
                             bdreg_block=1, bdreg_blocksize=2))]

    def test_avx512(self):
        f = self.forms(avx512())
        self.assertEqual("vbroadcastsd %%xmm3,%%zmm2", f[1])
        self.assertEqual("vshuff32x4 $0x55,%%zmm1,%%zmm1,%%zmm2", f[2])
        self.assertEqual("vinsertf32x4 $1,%%xmm1,%%zmm2,%%zmm2", f[4])

    def test_avx2(self):
        f = self.forms(fma256())
        self.assertEqual("vperm2f128 $0x11,%%ymm1,%%ymm1,%%ymm2", f[2])
        self.assertEqual("vextractf128 $1,%%ymm1,%%xmm2", f[3])

    @unittest.skipUnless(shutil.which("as") and shutil.which("aarch64-linux-gnu-as"),
                         "needs the host and AArch64 assemblers")
    def test_assemble(self):
        with tempfile.TemporaryDirectory() as d:
            for g in (avx512(), fma256()):
                src = os.path.join(d, "x.s")
                with open(src, "w") as f:
                    f.write("\n".join(l.replace("%%", "%") for l in self.forms(g)) + "\n")
                r = subprocess.run(["as", src, "-o", os.path.join(d, "x.o")],
                                   capture_output=True, text=True)
                self.assertEqual(0, r.returncode, r.stderr)


class test_sve_broadcast_offset(unittest.TestCase):

    def test_ld1r(self):
        # ld1rd z, p/z, [x, #imm]: 0..504 in multiples of 8
        g = sve()
        load = lambda off: plain(g.load(dregs=[g.vreg(1)], areg=g.greg(2), dt=adt.FP64,
                                        modifiers={mm.MASK, mm.IOFFSET},
                                        operand_modifiers={"adreg": {B}}, amreg=g.mreg(0),
                                        ioffset=off))
        self.assertEqual("ld1rd {z1.d}, p0/z, [x2, #504]", load(504))
        for bad in (512, 4):
            with self.assertRaises(Exception):
                load(bad)


class test_avx512_256bit_blocks(unittest.TestCase):

    def test_forms(self):
        a = avx512()
        v = a.vreg
        call = lambda **kw: plain(a.move(dregs=[v(1), v(2)], dts=[adt.FP64] * 2, **kw))
        lines = [
            call(operand_modifiers={"adreg": {omod.IBLOCK}, "bdreg": BLK}, adreg_block=1,
                 adreg_blocksize=4, bdreg_blocksize=4),
            call(operand_modifiers={"adreg": {omod.IBLOCK}}, adreg_block=1, adreg_blocksize=4),
            call(operand_modifiers={"bdreg": {omod.IBLOCK}}, bdreg_block=1, bdreg_blocksize=4)]
        self.assertEqual(["vshuff64x2 $0xee,%%zmm1,%%zmm1,%%zmm2",
                          "vextractf64x4 $1,%%zmm1,%%ymm2",
                          "vinsertf64x4 $1,%%ymm1,%%zmm2,%%zmm2"], lines)
        if shutil.which("as"):
            with tempfile.TemporaryDirectory() as d:
                src = os.path.join(d, "x.s")
                with open(src, "w") as f:
                    f.write("\n".join(l.replace("%%", "%") for l in lines) + "\n")
                r = subprocess.run(["as", src, "-o", os.path.join(d, "x.o")],
                                   capture_output=True, text=True)
                self.assertEqual(0, r.returncode, r.stderr)
