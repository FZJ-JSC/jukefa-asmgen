"""
opmem fixes ahead of the old vector loads and stores going onto load and
store (0.6, phase 5):

- gathers and scatters take byte offsets on every ISA - RVV can't scale
  them, AVX's use scale 1, greg_to_voffs builds them, the old methods took
  them - SVE's 32-bit ones were scaled by the element size (sxtw #2)
- SVE: 32-bit offsets in 64-bit lanes (sxtw); vector bases [z.d, #imm]
  printed as a scalar base's [x, #imm]
- RVV: an index narrower than the data (vluxei32.v under SEW 64), the
  mnemonic's EEW the index's
- AVX: FP64 broadcast into an xmm register is vmovddup (vbroadcastsd has no
  xmm destination)
"""
import unittest

from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.avx import fma128, fma256
from asmgen.registers import asm_data_type as adt, asm_index_type as ait
from asmgen.asmblocks.op.opmem import opmem_modifier as M
from asmgen.asmblocks.op.operand import operand_modifier as OM


class test_sve_gathers(unittest.TestCase):
    def setUp(self):
        self.g = sve()
        self.kw = dict(areg=self.g.greg(0), amreg=self.g.mreg(0), vidxreg=self.g.vreg(1))

    def gather(self, action, dt, it, mods=(), **kw):
        return getattr(self.g, action)(dregs=[self.g.vreg(0)], dt=dt, it=it,
                                       modifiers={M.MASK, M.VINDEX, *mods}, **self.kw, **kw).strip()

    def test_byte_offsets(self):
        self.assertEqual('"ld1d {z0.d}, p0/z, [x0, z1.d]\\n\\t"', self.gather("load", adt.FP64, ait.INT64))
        self.assertEqual('"ld1w {z0.s}, p0/z, [x0, z1.s, sxtw]\\n\\t"', self.gather("load", adt.FP32, ait.INT32))

    def test_32bit_offsets_in_64bit_lanes(self):
        self.assertEqual('"ld1d {z0.d}, p0/z, [x0, z1.d, sxtw]\\n\\t"', self.gather("load", adt.FP64, ait.INT32))
        self.assertEqual('"st1d {z0.d}, p0, [x0, z1.d, sxtw]\\n\\t"', self.gather("store", adt.FP64, ait.INT32))

    def test_vector_bases(self):
        self.assertEqual('"ld1d {z0.d}, p0/z, [z1.d, #8]\\n\\t"',
                         self.gather("load", adt.FP64, ait.INT64, mods={M.IOFFSET}, ioffset=8))


class test_rvv_index_widths(unittest.TestCase):
    def setUp(self):
        self.g = rvv()

    def indexed(self, action, dt, it):
        return getattr(self.g, action)(dregs=[self.g.vreg(0)], areg=self.g.greg(5), dt=dt, it=it,
                                       modifiers={M.VINDEX}, vidxreg=self.g.vreg(1)).strip()

    def test_eew_is_the_index_width(self):
        self.assertEqual('"vluxei32.v v0, (t5), v1\\n\\t"', self.indexed("load", adt.FP64, ait.INT32))
        self.assertEqual('"vluxei64.v v0, (t5), v1\\n\\t"', self.indexed("load", adt.FP64, ait.INT64))
        self.assertEqual('"vsuxei16.v v0, (t5), v1\\n\\t"', self.indexed("store", adt.FP32, ait.INT16))

    def test_no_index_wider_than_the_data(self):
        # it would take a register group
        with self.assertRaises(ValueError):
            self.indexed("load", adt.FP32, ait.INT64)


class test_avx_broadcasts(unittest.TestCase):
    def bcast(self, gen, dt):
        g = gen()
        return g.load(dregs=[g.vreg(0)], areg=g.greg(8), dt=dt,
                      operand_modifiers={"adreg": {OM.BCAST}}).strip()

    def test_xmm(self):
        self.assertEqual('"vmovddup (%%rax), %%xmm0\\n\\t"', self.bcast(fma128, adt.FP64))
        self.assertEqual('"vbroadcastss (%%rax), %%xmm0\\n\\t"', self.bcast(fma128, adt.FP32))

    def test_ymm_unchanged(self):
        self.assertEqual('"vbroadcastsd (%%rax), %%ymm0\\n\\t"', self.bcast(fma256, adt.FP64))


if __name__ == "__main__":
    unittest.main()
