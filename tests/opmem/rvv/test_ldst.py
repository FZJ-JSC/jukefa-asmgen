# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RVV opmem (ld/st) testsuite
"""
import unittest

from asmgen.asmblocks.op import opmem_modifier as mod
from asmgen.asmblocks.op import operand_modifier as opd_mod
from asmgen.registers import asm_data_type as adt, asm_index_type as ait
from asmgen.asmblocks.riscv64.types import riscv64_greg, riscv64_freg
from asmgen.asmblocks.rvv.types import rvv_vreg
from asmgen.asmblocks.rvv import rvv  # Updated import

class test_rvv_opmem(unittest.TestCase):
    """
    RVV opmem tests
    """
    def setUp(self):
        # We'll map these to standard indices (e.g., t0 is often t0)
        self.t0 = riscv64_greg(0)
        self.t1 = riscv64_greg(1)
        self.f0 = riscv64_freg(0)

        # Populate a list of 32 vector registers to allow nice slicing like self.vs[:1]
        self.vs = [rvv_vreg(i) for i in range(32)]

        # Using the wrapper class directly
        self.rvv = rvv()
        self.rvv.set_output_inline(False)

    def test_scalar_routing(self):
        """ Test that scalar registers are successfully routed through the rvv wrapper """
        self.assertEqual(
            self.rvv.load(dregs=[self.f0], areg=self.t0, dt=adt.FP32, modifiers=set()),
            "flw f0, 0(t0)\n" # Assuming riscv64_greg(5) prints as t0
        )

    def test_unit_stride(self):
        """ Test contiguous unit-stride vector loads/stores """
        self.assertEqual(
            self.rvv.load(dregs=self.vs[:1], areg=self.t0, dt=adt.FP64, modifiers=set()),
            "vle64.v v0, (t0)\n"
        )
        self.assertEqual(
            self.rvv.store(dregs=self.vs[:1], areg=self.t0, dt=adt.FP32, modifiers=set()),
            "vse32.v v0, (t0)\n"
        )

    def test_strided_vector_operations(self):
        """ Test vector load/store with GSTRIDE (vlse/vsse) """
        self.assertEqual(
            self.rvv.load(dregs=self.vs[:1], areg=self.t0, dt=adt.FP64,
                          modifiers={mod.GSTRIDE}, streg=self.t1),
            "vlse64.v v0, (t0), t1\n"
        )

    def test_broadcast_load(self):
        """
        Test broadcasting a single element to all lanes
        (BCAST mapped to vlse with zero stride)
        """
        self.assertEqual(
            self.rvv.load(dregs=self.vs[:1], areg=self.t0, dt=adt.FP32,
                          modifiers=set(),
                          operand_modifiers={'adreg': {opd_mod.BCAST}}),
            "vlse32.v v0, (t0), zero\n"
        )

    def test_broadcast_segmented_load(self):
        """ Test broadcasting a structure to all lanes (vlseg with zero stride) """
        # e.g., Broadcasting an RGB pixel to all lanes
        self.assertEqual(
            self.rvv.load(dregs=self.vs[4:7], areg=self.t0, dt=adt.UINT8,
                          modifiers={mod.STRUCT}, nstructs=3,
                          operand_modifiers={'adreg': {opd_mod.BCAST},
                                             'bdreg': {opd_mod.BCAST},
                                             'cdreg': {opd_mod.BCAST}}),
            "vlsseg3e8.v v4, (t0), zero\n"
        )

    def test_bcast_on_store_fails(self):
        """ Test that trying to broadcast on a store is correctly rejected """
        with self.assertRaisesRegex(ValueError, "only valid for LOAD"):
            self.rvv.store(dregs=self.vs[:1], areg=self.t0, dt=adt.FP32,
                           modifiers=set(),
                           operand_modifiers={'adreg': {opd_mod.BCAST}})

    def test_indexed_vector_operations(self):
        """ Test unordered indexed vector operations (vluxei/vsuxei) """
        self.assertEqual(
            self.rvv.store(dregs=self.vs[:1], areg=self.t0,
                           dt=adt.FP32, it=ait.INT32,
                           modifiers={mod.VINDEX}, vidxreg=self.vs[2]),
            "vsuxei32.v v0, (t0), v2\n"
        )

    def test_structured_segment_operations(self):
        """ Test segment load operations (vlseg) """
        self.assertEqual(
            self.rvv.load(dregs=self.vs[:2], areg=self.t0, dt=adt.FP64,
                          modifiers={mod.STRUCT}, nstructs=2),
            "vlseg2e64.v v0, (t0)\n"
        )

    def test_mutually_exclusive_modifiers(self):
        """ Test that GSTRIDE and VINDEX are rejected together """
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.rvv.load(dregs=self.vs[:1], areg=self.t0, dt=adt.FP32,
                          modifiers={mod.GSTRIDE, mod.VINDEX}, streg=self.t1, vidxreg=self.vs[2])

    def test_lmul_contiguity(self):
        """ Test that passing non-contiguous segment registers raises an error """
        # Assuming default LMUL=1, passing v0 and v2 should fail
        with self.assertRaisesRegex(ValueError,
                                    "index of bdreg must be index of adreg plus 1"):
            self.rvv.load(dregs=[self.vs[0], self.vs[2]], areg=self.t0, dt=adt.FP32,
                          modifiers={mod.STRUCT}, nstructs=2)

    def test_missing_required_parameters(self):
        """ Test kwargs validation for VINDEX and STRUCT """
        with self.assertRaisesRegex(ValueError,
                                    "VINDEX modifier requires '(vidxreg|it)' parameter"):
            self.rvv.load(dregs=self.vs[:1], areg=self.t0, dt=adt.FP32,
                          modifiers={mod.VINDEX})

        with self.assertRaisesRegex(ValueError,
                                    "STRUCT modifier requires 'nstructs' parameter"):
            self.rvv.load(dregs=self.vs[:1], areg=self.t0, dt=adt.FP32, modifiers={mod.STRUCT})


class test_strided_segments(unittest.TestCase):
    """
    Strided segments (STRUCT and GSTRIDE): nf registers' elements i stored
    together, segment i streg bytes from segment i-1 - a panel's rows packed
    in parts, each part's rows interleaved into columns cdim apart; at most
    8 registers (EMUL * NF), fields a register group apart
    """
    def gen(self, lmul=1):
        from asmgen.asmblocks.rvv import rvv
        g = rvv()
        g.set_output_inline(False)
        g.set_parameter("LMUL", lmul)
        return g

    def call(self, g, action, regs, n):
        kw = dict(dregs=regs, areg=g.greg(2), dt=adt.FP64, modifiers={mod.STRUCT, mod.GSTRIDE}, nstructs=n,
                  streg=g.greg(5))
        return getattr(g, action), kw

    def test_forms(self):
        g = self.gen()
        op, kw = self.call(g, "store", [g.vreg(i) for i in (8, 9, 10)], 3)
        self.assertEqual("vssseg3e64.v v8, (t2), t5", op(**kw).strip())
        op, kw = self.call(g, "load", [g.vreg(i) for i in (8, 9, 10)], 3)
        self.assertEqual("vlsseg3e64.v v8, (t2), t5", op(**kw).strip())

    def test_lmul(self):
        # EMUL * NF <= 8, the signatures rebuilt where LMUL changes: at m2 four
        # fields, two registers apart (vreg counts groups: vreg(8) is v16)
        from asmgen.asmblocks.op.opmem import opmem_modifier as M
        for lmul, most in ((1, 8), (2, 4), (4, 2)):
            g = self.gen(lmul)
            with self.subTest(lmul=lmul):
                self.assertEqual(most, max(f.structural_params["nstructs"] for f in
                                           g.store.forms(modifiers={M.STRUCT, M.GSTRIDE}, dts={"adreg": adt.FP64})))
                self.assertEqual(most, max(f.structural_params["nstructs"] for f in
                                           g.store.forms(modifiers={M.STRUCT}, dts={"adreg": adt.FP64})))
        g = self.gen(2)
        op, kw = self.call(g, "store", [g.vreg(8), g.vreg(9)], 2)
        self.assertEqual("vssseg2e64.v v16, (t2), t5", op(**kw).strip())
        op, kw = self.call(g, "store", [g.vreg(8), g.vreg(10)], 2)
        self.assertIsNone(op.fits(**kw))

    def test_semantics(self):
        # run: v8-v10 hold three rows, stored as segments 5 doubles apart:
        # element 5i + p is row p's element i (qemu, VLEN 128 and 256)
        import os, shutil, subprocess, tempfile
        cc, qemu = "riscv64-linux-gnu-gcc", "qemu-riscv64-static"
        if not (shutil.which(cc) and shutil.which(qemu)):
            self.skipTest("needs riscv64-linux-gnu-gcc and qemu-riscv64-static")
        g = self.gen()
        op, kw = self.call(g, "store", [g.vreg(i) for i in (8, 9, 10)], 3)
        kw.update(areg=g.greg(19), streg=g.greg(3))       # a1: out, t3: the stride
        asm = ("\t.text\n\t.globl f\n\t.type f, @function\nf:\n"
               "\tvsetvli t0, zero, e64, m1, ta, ma\n"
               "\tvle64.v v8, (a0)\n\taddi a0, a0, 256\n\tvle64.v v9, (a0)\n\taddi a0, a0, 256\n"
               "\tvle64.v v10, (a0)\n\tli t3, 40\n" + op(**kw) + "\tmv a0, t0\n\tret\n")
        c = r"""
#include <stdio.h>
long f(const double* rows, double* out);
int main(void) {
    double rows[3 * 32], out[5 * 32];
    for (int i = 0; i < 3 * 32; ++i) rows[i] = i;
    for (int i = 0; i < 5 * 32; ++i) out[i] = -1;
    long vl = f(rows, out); int bad = 0;
    for (long i = 0; i < vl; ++i) for (int p = 0; p < 3; ++p) if (out[5 * i + p] != rows[32 * p + i]) ++bad;
    for (long i = 0; i < vl; ++i) for (int p = 3; p < 5; ++p) if (out[5 * i + p] != -1) ++bad;
    printf("vl %ld bad %d\n", vl, bad);
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "f.s"), "w").write(asm)
            open(os.path.join(d, "m.c"), "w").write(c)
            r = subprocess.run([cc, "-static", "-march=rv64gcv", os.path.join(d, "m.c"), os.path.join(d, "f.s"),
                                "-o", os.path.join(d, "run")], capture_output=True, text=True)
            self.assertEqual(0, r.returncode, r.stderr)
            for vlen in (128, 256):
                out = subprocess.run([qemu, "-cpu", f"rv64,v=true,vlen={vlen},elen=64", os.path.join(d, "run")],
                                     capture_output=True, text=True, timeout=60).stdout
                self.assertIn(f"vl {vlen // 64} bad 0", out, out)


if __name__ == '__main__':
    unittest.main()
