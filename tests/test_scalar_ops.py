"""
Operations on scalar registers across ISAs: general registers moved by a
shifted register in one instruction, RISC-V's FP formats in zeroing and
copying FP registers
"""

import unittest

from asmgen.asmblocks.sme import sme
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.op.opset import opset_modifier
from asmgen.asmblocks.op.operand import operand_modifier
from asmgen.asmblocks.op.opmem import opmem_modifier


class test_shifted_adds(unittest.TestCase):
    """
    dst = reg1 +/- (reg2 << shift) in one instruction where the ISA has one;
    NotImplementedError where it hasn't
    """

    def test_aarch64(self):
        g = sme()
        g.output_inline = False
        self.assertEqual("add x0,x0,x9, LSL #1",
                         g.add(adreg=g.greg(0), bdreg=g.greg(9), cdreg=g.greg(0), a_dt=adt.UINT64, b_dt=adt.UINT64, c_dt=adt.UINT64, operand_modifiers={"bdreg": {operand_modifier.ISHIFT}}, bdreg_shift=1).strip())
        self.assertEqual("sub x0,x0,x9, LSL #2",
                         g.sub(adreg=g.greg(0), bdreg=g.greg(9), cdreg=g.greg(0), a_dt=adt.UINT64, b_dt=adt.UINT64, c_dt=adt.UINT64, operand_modifiers={"bdreg": {operand_modifier.ISHIFT}}, bdreg_shift=2).strip())

    def test_x86_and_riscv(self):
        from asmgen.asmblocks.avx.avx2 import fma256
        from asmgen.asmblocks.rvv import rvv
        g = fma256()
        g.output_inline = False
        self.assertIn(",4)", g.add(adreg=g.greg(0), bdreg=g.greg(9), cdreg=g.greg(0), a_dt=adt.UINT64, b_dt=adt.UINT64, c_dt=adt.UINT64, operand_modifiers={"bdreg": {operand_modifier.ISHIFT}}, bdreg_shift=2))
        with self.assertRaises(ValueError):
            g.add(adreg=g.greg(0), bdreg=g.greg(9), cdreg=g.greg(0), a_dt=adt.UINT64, b_dt=adt.UINT64, c_dt=adt.UINT64, operand_modifiers={"bdreg": {operand_modifier.ISHIFT}}, bdreg_shift=4)
        with self.assertRaises(ValueError):
            g.sub(adreg=g.greg(0), bdreg=g.greg(9), cdreg=g.greg(0), a_dt=adt.UINT64, b_dt=adt.UINT64, c_dt=adt.UINT64, operand_modifiers={"bdreg": {operand_modifier.ISHIFT}}, bdreg_shift=1)
        r = rvv()
        r.set_features([])          # the baseline ISA: no Zba, no shifted add
        with self.assertRaises(ValueError):
            r.add(adreg=r.greg(0), bdreg=r.greg(9), cdreg=r.greg(0), a_dt=adt.UINT64, b_dt=adt.UINT64, c_dt=adt.UINT64, operand_modifiers={"bdreg": {operand_modifier.ISHIFT}}, bdreg_shift=1)
        r.set_features(["ZBA"])
        self.assertIn("sh1add", r.add(adreg=r.greg(0), bdreg=r.greg(9), cdreg=r.greg(0), a_dt=adt.UINT64, b_dt=adt.UINT64, c_dt=adt.UINT64, operand_modifiers={"bdreg": {operand_modifier.ISHIFT}}, bdreg_shift=1))


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
            self.assertEqual(f"fcvt.{suf}.w f29, zero", g.set(dreg=g.freg(29, dt), dt=dt, modifiers={opset_modifier.CONST}, constant="zero").strip())
            self.assertEqual(f"fmv.{suf} f2,f1", g.move(dregs=[g.freg(1, dt), g.freg(2, dt)], dts=[dt, dt]).strip())
        # loads keep w
        self.assertEqual(f"flw f3, 8({g.greg(10)})",
                         g.load(dregs=[g.freg(3, adt.FP32)], areg=g.greg(10), dt=adt.FP32, modifiers={opmem_modifier.IOFFSET}, ioffset=8).strip())


if __name__ == "__main__":
    unittest.main()


class test_avx512_fp_zero(unittest.TestCase):
    """
    AVX-512 keeps FP scalars in the vector file, allocated from the top:
    zeroing one in xmm16-31 needs EVEX - vpxord, not VEX's vpxor
    """

    def test_evex(self):
        import shutil
        import subprocess
        import tempfile
        from asmgen.asmblocks.avx.avx512 import avx512
        g = avx512()
        g.output_inline = False
        line = g.set(dreg=g.freg(20, adt.FP64), dt=adt.FP64, modifiers={opset_modifier.CONST}, constant="zero").strip()
        self.assertEqual("vpxord %xmm20,%xmm20,%xmm20", line)
        if shutil.which("as"):
            with tempfile.TemporaryDirectory() as d:
                with open(f"{d}/z.s", "w") as f:
                    f.write(line + "\n")
                r = subprocess.run(["as", f"{d}/z.s", "-o", f"{d}/z.o"], capture_output=True, text=True)
                self.assertEqual(0, r.returncode, r.stderr)


class test_scaled_offset(unittest.TestCase):
    """
    SCALEOFFSET: an address as base + index register x scale_factor (1, 2,
    4, 8) - signatures carrying the scale as a structural parameter, the
    forms assembling
    """

    def test_x86(self):
        import shutil
        import subprocess
        import tempfile
        from asmgen.asmblocks.avx.avx512 import avx512
        from asmgen.asmblocks.op import opmem_modifier as mod
        g = avx512()
        g.output_inline = False
        scales = sorted({s.structural_params.get("scale_factor") for s in g.load.get_signatures()
                         if mod.SCALEOFFSET in s.modifiers})
        self.assertEqual([1, 2, 4, 8], scales)
        lines = []
        for scale in scales:
            line = g.load(dregs=[g.vreg(3)], areg=g.greg(4), dt=adt.FP64,
                          modifiers={mod.GOFFSET, mod.SCALEOFFSET}, offreg=g.greg(5),
                          scale_factor=scale).strip()
            self.assertIn(f",{scale})", line)
            lines.append(line)
        lines.append(g.load(dregs=[g.vreg(3)], areg=g.greg(4), dt=adt.FP64,
                            modifiers={mod.GOFFSET, mod.SCALEOFFSET, mod.IOFFSET}, offreg=g.greg(5),
                            scale_factor=4, ioffset=64).strip())
        self.assertTrue(lines[-1].startswith("vmovupd 64("), lines[-1])
        # a broadcast through a scaled index
        from asmgen.asmblocks.op.operand import operand_modifier as om
        lines.append(g.load(dregs=[g.vreg(3)], areg=g.greg(4), dt=adt.FP64,
                            modifiers={mod.GOFFSET, mod.SCALEOFFSET}, offreg=g.greg(5), scale_factor=2,
                            operand_modifiers={"adreg": {om.BCAST}}).strip())
        self.assertIn(",2)", lines[-1])
        if shutil.which("as"):
            with tempfile.TemporaryDirectory() as d:
                with open(f"{d}/s.s", "w") as f:
                    f.write("\n".join(lines) + "\n")
                r = subprocess.run(["as", f"{d}/s.s", "-o", f"{d}/s.o"], capture_output=True, text=True)
                self.assertEqual(0, r.returncode, r.stderr)
