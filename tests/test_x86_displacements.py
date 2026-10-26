"""
x86 displacements (0.6, phase 5): offsets in disp8's range by default -
one signed byte under VEX, the multiples of the access's size N up to
127*N under EVEX (disp8*N) - the full signed 32 bits with the encoding
feature enable-disp32, no offsets at all with neither. set_features (ISA
extensions) leaves both alone.
"""
import unittest

from asmgen.asmblocks.avx import fma128, fma256, avx512
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.op.operand import operand_modifier
from asmgen.asmblocks.op.opmem import opmem_modifier


def loads(g, offset=0):
    """a vector load at an immediate offset, as kernels write it"""
    kw = {"modifiers": {opmem_modifier.IOFFSET}, "ioffset": offset} if offset else {}
    return g.load(dregs=[g.vreg(0)], areg=g.greg(8), dt=adt.FP64, **kw)


class test_defaults(unittest.TestCase):
    def test_parameters(self):
        for gen in (fma128, fma256, avx512):
            g = gen()
            self.assertIn("enable-disp8", g.get_parameters())
            self.assertEqual("on", g.get_param_value("enable-disp8"))
            self.assertEqual("off", g.get_param_value("enable-disp32"))

    def test_vex_one_byte(self):
        g = fma256()
        for offset in (-128, -1, 1, 127):
            loads(g, offset=offset)
        for offset in (-129, 128, 4096):
            with self.assertRaises(ValueError):
                loads(g, offset=offset)

    def test_vex_vectors(self):
        # 3 ymm, 7 xmm vectors stay within 127 bytes
        fma256().load(dregs=[fma256().vreg(0)], areg=fma256().greg(8), dt=adt.FP64, modifiers={opmem_modifier.VOFFSET}, voffset=3)
        with self.assertRaises(ValueError):
            g = fma256()
            g.load(dregs=[g.vreg(0)], areg=g.greg(8), dt=adt.FP64, modifiers={opmem_modifier.VOFFSET}, voffset=4)
        g = fma128()
        g.load(dregs=[g.vreg(0)], areg=g.greg(8), dt=adt.FP64, modifiers={opmem_modifier.VOFFSET}, voffset=7)
        with self.assertRaises(ValueError):
            g.load(dregs=[g.vreg(0)], areg=g.greg(8), dt=adt.FP64, modifiers={opmem_modifier.VOFFSET}, voffset=8)

    def test_evex_scaled(self):
        g = avx512()
        for offset in (-128 * 64, 64, 127 * 64):
            loads(g, offset=offset)
        for offset in (8, 128 * 64, -129 * 64):      # not a multiple of 64, or past 127*64
            with self.assertRaises(ValueError):
                loads(g, offset=offset)
        g.load(dregs=[g.vreg(0)], areg=g.greg(8), dt=adt.FP64, modifiers={opmem_modifier.VOFFSET}, voffset=127)
        with self.assertRaises(ValueError):
            g.load(dregs=[g.vreg(0)], areg=g.greg(8), dt=adt.FP64, modifiers={opmem_modifier.VOFFSET}, voffset=128)

    def test_evex_broadcasts_scale_by_the_element(self):
        g = avx512()
        g.load(dregs=[g.vreg(0)], areg=g.greg(8), dt=adt.FP64, modifiers={opmem_modifier.IOFFSET}, ioffset=127 * 8, operand_modifiers={"adreg": {operand_modifier.BCAST}})
        for offset in (4, 128 * 8):
            with self.assertRaises(ValueError):
                g.load(dregs=[g.vreg(0)], areg=g.greg(8), dt=adt.FP64, modifiers={opmem_modifier.IOFFSET}, ioffset=offset, operand_modifiers={"adreg": {operand_modifier.BCAST}})

    def test_scalar(self):
        g = fma256()
        g.load(dregs=[g.greg(9)], areg=g.greg(8), dt=adt.UINT64, modifiers={opmem_modifier.IOFFSET}, ioffset=127)
        with self.assertRaises(ValueError):
            g.load(dregs=[g.greg(9)], areg=g.greg(8), dt=adt.UINT64, modifiers={opmem_modifier.IOFFSET}, ioffset=128)


class test_switches(unittest.TestCase):
    def test_disp32(self):
        for gen, offset in ((fma256, 4096), (avx512, 8), (avx512, 2**31 - 1)):
            g = gen()
            g.set_parameter("enable-disp32", "on")
            self.assertIn(f"{offset}(", loads(g, offset=offset))
        g = fma256()
        g.set_parameter("enable-disp32", "on")
        g.load(dregs=[g.greg(9)], areg=g.greg(8), dt=adt.UINT64, modifiers={opmem_modifier.IOFFSET}, ioffset=-2**31)

    def test_no_offsets(self):
        g = fma256()
        g.set_parameter("enable-disp8", "off")
        with self.assertRaises(ValueError):
            loads(g, offset=8)
        self.assertIn("(%", loads(g, offset=0))        # offset 0: the plain form

    def test_set_features_leaves_them(self):
        g = avx512()
        g.set_features([])
        loads(g, offset=64)
        with self.assertRaises(ValueError):
            loads(g, offset=8)
        g.set_parameter("enable-disp32", "on")
        g.set_features(["VL"])
        loads(g, offset=8)


if __name__ == "__main__":
    unittest.main()
