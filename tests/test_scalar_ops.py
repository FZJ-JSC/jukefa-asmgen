"""
Operations on scalar registers across ISAs: general registers moved by a
shifted register in one instruction, RISC-V's FP formats in zeroing and
copying FP registers
"""

import unittest

from asmgen.asmblocks.sme import sme
from asmgen.registers import asm_data_type as adt


class test_shifted_adds(unittest.TestCase):
    """
    dst = reg1 +/- (reg2 << shift) in one instruction where the ISA has one;
    NotImplementedError where it hasn't
    """

    def test_aarch64(self):
        g = sme()
        g.output_inline = False
        self.assertEqual("add x0,x0,x9, LSL #1",
                         g.add_greg_greg_shifted(dst=g.greg(0), reg1=g.greg(0), reg2=g.greg(9), shift=1).strip())
        self.assertEqual("sub x0,x0,x9, LSL #2",
                         g.sub_greg_greg_shifted(dst=g.greg(0), reg1=g.greg(0), reg2=g.greg(9), shift=2).strip())

    def test_x86_and_riscv(self):
        from asmgen.asmblocks.avx.avx2 import fma256
        from asmgen.asmblocks.rvv import rvv
        g = fma256()
        g.output_inline = False
        self.assertIn(",4)", g.add_greg_greg_shifted(dst=g.greg(0), reg1=g.greg(0), reg2=g.greg(9), shift=2))
        with self.assertRaises(NotImplementedError):
            g.add_greg_greg_shifted(dst=g.greg(0), reg1=g.greg(0), reg2=g.greg(9), shift=4)
        with self.assertRaises(NotImplementedError):
            g.sub_greg_greg_shifted(dst=g.greg(0), reg1=g.greg(0), reg2=g.greg(9), shift=1)
        r = rvv()
        with self.assertRaises(NotImplementedError):
            r.add_greg_greg_shifted(dst=r.greg(0), reg1=r.greg(0), reg2=r.greg(9), shift=1)


class test_riscv_fp_suffixes(unittest.TestCase):
    """
    RISC-V spells single precision w in loads, stores and moves to and from
    integer registers (flw, fmv.w.x), s in arithmetic and conversions
    (fcvt.s.w, fmv.s): zeroing and copying FP registers take the latter
    """

    def test_single_and_double(self):
        from asmgen.asmblocks.rvv import rvv
        g = rvv()
        g.output_inline = False
        for dt, suf in ((adt.FP32, "s"), (adt.FP64, "d")):
            self.assertEqual(f"fcvt.{suf}.w f29, zero", g.zero_freg(freg=g.freg(29, dt), dt=dt).strip())
            self.assertEqual(f"fmv.{suf} f2,f1", g.mov_freg(src=g.freg(1, dt), dst=g.freg(2, dt), dt=dt).strip())
        # loads keep w
        self.assertEqual(f"flw f3, 8({g.greg(10)})",
                         g.load_scalar_immoff(areg=g.greg(10), offset=8, freg=g.freg(3, adt.FP32), dt=adt.FP32).strip())


if __name__ == "__main__":
    unittest.main()
