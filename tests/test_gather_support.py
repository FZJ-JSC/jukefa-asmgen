"""
Gather and scatter support: masks set all-true, index vectors, the vector length
"""

import unittest

from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.avx import avx512
from asmgen.asmblocks.avx.avx2 import fma256
from asmgen.registers import asm_data_type as adt


def plain(t):
    return t.strip().strip('"').replace("\\n", "").replace("\\t", "").replace("%%", "%").strip()


class test_gather_support(unittest.TestCase):

    def test_mask_all_true(self):
        a, f, s = avx512(), fma256(), sve()
        self.assertEqual("kxnorw %k1,%k1,%k1", plain(a.mask_all_true(reg=a.mreg(1), dt=adt.FP64)))
        self.assertEqual("vpcmpeqd %ymm2,%ymm2,%ymm2", plain(f.mask_all_true(reg=f.vreg(2), dt=adt.FP64)))
        self.assertIn("ptrue p1.d", plain(s.mask_all_true(reg=s.mreg(1), dt=adt.FP64)))

    def test_avx2_index_vector(self):
        # vpmullq is AVX-512: AVX2 multiplies the low dwords
        f = fma256()
        text = f.greg_to_voffs(streg=f.greg(3), vreg=f.vreg(4), dt=adt.FP64)
        self.assertIn("vpmuludq", text)
        self.assertNotIn("vpmullq", text)

    def test_byte_offsets(self):
        # index vectors hold byte offsets on every ISA: AVX gathers scale by 1
        a = avx512()
        sig = next(s for s in a.load.get_signatures()
                   if "VINDEX" in {m.name for m in s.modifiers}
                   and s.operands["adreg"].dt == adt.FP64)
        from asmgen.asmblocks.op import opmem_modifier as mm
        text = plain(a.load(dregs=[a.vreg(4)], areg=a.greg(8), dt=adt.FP64,
                            modifiers=set(sig.modifiers), vidxreg=a.vreg(5), amreg=a.mreg(1),
                            **sig.structural_params))
        self.assertIn(",%zmm5,1)", text)

    def test_vlen_to_greg(self):
        s = sve()
        self.assertEqual("cntd x8", plain(s.vlen_to_greg(reg=s.greg(8), dt=adt.FP64)))
        with self.assertRaises(NotImplementedError):
            avx512().vlen_to_greg(reg=avx512().greg(8), dt=adt.FP64)


if __name__ == "__main__":
    unittest.main()
