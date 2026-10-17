# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Test AVX loads/stores
"""
import unittest

from asmgen.asmblocks.op import (
    opmem_modifier as mod,
    operand_modifier as opd_mod
)
from asmgen.registers import asm_data_type as adt, asm_index_type as ait
from asmgen.asmblocks.avx.types import (
    x86_greg,
    xmm_vreg, ymm_vreg, zmm_vreg,
    avx_freg,
    avx512_mreg,
    reg_prefixer,
)
from asmgen.asmblocks.avx.opmem import avx128_load, avx256_load, avx512_load
from asmgen.asmblocks.avx.opmem import avx128_store, avx512_store

def asmwrap(s: str) -> str:
    """
    Dummy asmwrap
    """
    lines = s.split("\n")
    return "".join(f"{line}\n" for line in lines)

# fine for testing
# pylint: disable-next=too-many-instance-attributes
class test_avx_opmem(unittest.TestCase):
    """
    Testsuite for AVX2 128/256 bit + AVX512 opmem operations
    """
    def setUp(self):
        self.r8 = x86_greg(0)
        self.r15 = x86_greg(7)

        self.xmm0 = xmm_vreg(0)
        self.ymm0 = ymm_vreg(self.xmm0.idx)
        self.zmm0 = zmm_vreg(self.xmm0.idx)
        self.xmm1 = xmm_vreg(1)
        self.ymm1 = ymm_vreg(self.xmm1.idx)
        self.zmm1 = zmm_vreg(self.xmm1.idx)
        self.xmm2 = xmm_vreg(2)
        self.ymm2 = ymm_vreg(self.xmm2.idx)
        self.f0   = avx_freg(0) # scalar float
        self.k2   = avx512_mreg(2)

        self.output_inline = False
        self.asmwrap = asmwrap
        self.rpref = reg_prefixer(lambda : False)

        self.load_128  = avx128_load(asmwrap=self.asmwrap, rpref=self.rpref)
        self.load_256  = avx256_load(asmwrap=self.asmwrap, rpref=self.rpref)
        self.load_512  = avx512_load(asmwrap=self.asmwrap, rpref=self.rpref)
        self.store_128 = avx128_store(asmwrap=self.asmwrap, rpref=self.rpref)
        self.store_512 = avx512_store(asmwrap=self.asmwrap, rpref=self.rpref)

    # --- 1. Basic Routing & Unit Stride ---
    def test_scalar_routing(self):
        """
        Ensure scalar registers fall back to x86_opmem
        """
        self.assertEqual(
            self.load_128(dregs=[self.f0], areg=self.r8,
                          dt=adt.FP32, modifiers=set()),
            "vmovss (%r8), %xmm0\n"
        )

    def test_basic_vector_loads(self):
        """ Test standard vmovups / vmovupd """
        self.assertEqual(
            self.load_128(dregs=[self.xmm0], areg=self.r8,
                          dt=adt.FP32, modifiers=set()),
            "vmovups (%r8), %xmm0\n"
        )
        self.assertEqual(
            self.load_512(dregs=[self.zmm1], areg=self.r15,
                          dt=adt.FP64, modifiers=set()),
            "vmovupd (%r15), %zmm1\n"
        )

    def test_vector_offsets(self):
        """ VOFFSET scales by the SIMD byte size (16, 32, 64) """
        self.assertEqual(
            self.load_128(dregs=[self.xmm0], areg=self.r8, dt=adt.FP32,
                          modifiers={mod.VOFFSET}, voffset=2),
            "vmovups 32(%r8), %xmm0\n"
        )
        self.assertEqual(
            self.load_512(dregs=[self.zmm0], areg=self.r8, dt=adt.FP32,
                          modifiers={mod.VOFFSET}, voffset=2),
            "vmovups 128(%r8), %zmm0\n"
        )

    # --- 2. Broadcasts ---
    def test_broadcasts(self):
        """
        Test broadcasts (vbroadcastsx)
        """
        self.assertEqual(
            self.load_256(dregs=[self.ymm0], areg=self.r8, dt=adt.FP32,
                          operand_modifiers={'adreg':{opd_mod.BCAST}}),
            "vbroadcastss (%r8), %ymm0\n"
        )
        # BCAST should fail on STORE
        with self.assertRaisesRegex(ValueError, "BCAST modifier can't be used with stores"):
            self.store_512(dregs=[self.zmm0], areg=self.r8, dt=adt.FP32,
                          operand_modifiers={'adreg':{opd_mod.BCAST}})

    # --- 3. Lane Loads (Primitives) ---
    def test_lane_loads_fp64(self):
        """
        Test FP64 vmovsd (lane 0) and vmovhpd (lane 1)
        """
        self.assertEqual(
            self.load_128(dregs=[self.xmm0], areg=self.r8, dt=adt.FP64,
                          operand_modifiers={'adreg':{opd_mod.ILANE}}, adreg_lane=0),
            "vmovsd (%r8), %xmm0\n"
        )
        self.assertEqual(
            self.load_128(dregs=[self.xmm0], areg=self.r8, dt=adt.FP64,
                          operand_modifiers={'adreg':{opd_mod.ILANE}}, adreg_lane=1),
            "vmovhpd (%r8), %xmm0, %xmm0\n"
        )

    def test_lane_loads_fp32(self):
        """
        Test FP32 vmovss (lane 0) and vinsertps (lane > 0)
        """
        self.assertEqual(
            self.load_128(dregs=[self.xmm1], areg=self.r8, dt=adt.FP32,
                          operand_modifiers={'adreg':{opd_mod.ILANE}}, adreg_lane=0),
            "vmovss (%r8), %xmm1\n"
        )
        # lane 1 << 4 = 0x10
        self.assertEqual(
            self.load_128(dregs=[self.xmm1], areg=self.r8, dt=adt.FP32,
                          operand_modifiers={'adreg':{opd_mod.ILANE}}, adreg_lane=1),
            "vinsertps $16, (%r8), %xmm1, %xmm1\n"
        )
    def test_lane_stores_float(self):
        """
        Test floating point lane memory stores
        """
        # FP64 lane 1
        self.assertEqual(
            self.store_128(dregs=[self.xmm0], areg=self.r8, dt=adt.FP64,
                           operand_modifiers={'adreg':{opd_mod.ILANE}}, adreg_lane=1),
            "vmovhpd %xmm0,(%r8)\n"
        )
        # FP32 lane 2 (2 << 4 = 32)
        self.assertEqual(
            self.store_128(dregs=[self.xmm1], areg=self.r8, dt=adt.FP32,
                           operand_modifiers={'adreg':{opd_mod.ILANE}}, adreg_lane=2),
            "vextractps $32, %xmm1, (%r8)\n"
        )
    def test_lane_ldst_integer(self):
        """
        Test Integer domain lane inserts/extracts
        """
        # 32-bit INT load (vpinsrd)
        self.assertEqual(
            self.load_128(dregs=[self.xmm0], areg=self.r8, dt=adt.SINT32,
                          operand_modifiers={'adreg':{opd_mod.ILANE}}, adreg_lane=3),
            "vpinsrd $3, (%r8), %xmm0, %xmm0\n"
        )
        # 8-bit INT store (vpextrb)
        self.assertEqual(
            self.store_128(dregs=[self.xmm2], areg=self.r8, dt=adt.UINT8,
                           operand_modifiers={'adreg':{opd_mod.ILANE}}, adreg_lane=15),
            "vpextrb $15, %xmm2, (%r8)\n"
        )

    # --- 4. Gather & Scatter (VINDEX) ---
    def test_gather_avx2(self):
        """
        AVX2 Gather format
        """
        self.assertEqual(
            self.load_256(dregs=[self.ymm0], amreg=self.ymm2, areg=self.r8,
                          dt=adt.FP64,
                          modifiers={mod.VINDEX, mod.MASK}, vidxreg=self.ymm1, it=ait.INT64),
            "vgatherqpd %ymm2, (%r8,%ymm1,1), %ymm0\n"
        )

    def test_gather_avx512(self):
        """
        AVX512 Gather with K-mask
        """
        expected = "vgatherdps (%r8,%zmm1,1), %zmm0{%k2}\n"
        self.assertEqual(
            self.load_512(dregs=[self.zmm0], amreg=self.k2, areg=self.r8,
                          dt=adt.FP32, modifiers={mod.VINDEX, mod.MASK},
                          vidxreg=self.zmm1, it=ait.INT32),
            expected
        )
    def test_scatter_avx512(self):
        """
        AVX512 Scatter with K-mask
        """
        expected = "vscatterdps %zmm0, (%r8,%zmm1,1){%k2}\n"
        self.assertEqual(
            self.store_512(dregs=[self.zmm0], amreg=self.k2, areg=self.r8,
                           dt=adt.FP32, modifiers={mod.VINDEX, mod.MASK},
                           vidxreg=self.zmm1, it=ait.INT32),
            expected
        )

    def test_scatter_unsupported(self):
        """
        AVX128/256 should reject scatter
        """
        with self.assertRaisesRegex(ValueError,
                                    "VINDEX modifier can't be used with avx2 128/256 bit stores"):
            store_128 = avx128_store(asmwrap=self.asmwrap,
                                     rpref=self.rpref)
            store_128(dregs=[self.xmm0], amreg=self.xmm2, areg=self.r8,
                      dt=adt.FP32, modifiers={mod.VINDEX, mod.MASK},
                      vidxreg=self.xmm1, it=ait.INT32)

if __name__ == '__main__':
    unittest.main()
