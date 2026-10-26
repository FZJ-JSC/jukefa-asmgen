"""
rvv071's operations print RVV 0.7.1's forms (0.6, phase 5) - th.-prefixed
under XTheadVector (on by default), unprefixed without it: loads and stores
with the element width as a letter (vle.v, vlw.v), strides and broadcasts
vls*.v, indices vlx*.v and vsux*.v at the data's width, segments
vlseg<nf>*.v; element 0 into a general purpose register vext.x.v; no FP
slides; vmpopc.m; vsetvli without policies. Checked against binutils'
XTheadVector. The forms below are XTheadVector's off: unprefixed.
"""
import os
import shutil
import subprocess
import tempfile
import unittest

from asmgen.asmblocks.rvv071 import rvv071
from asmgen.asmblocks.op.opmem import opmem_modifier as M
from asmgen.asmblocks.op.operand import operand_modifier as OM
from asmgen.registers import asm_data_type as adt, asm_index_type as ait
from asmgen.asmblocks.op.opset import opset_modifier
from asmgen.asmblocks.op.opmem import opmem_modifier


def plain(text):
    """the instruction of an inline asm string"""
    return text.strip().strip('"').replace("\\n\\t", "").strip()


class test_rvv071_memory(unittest.TestCase):
    def setUp(self):
        self.g = rvv071()
        self.g.set_parameter("XTheadVector", "off")
        self.v0, self.v1 = self.g.vreg(0), self.g.vreg(1)
        self.t0, self.t1 = self.g.greg(5), self.g.greg(6)

    def access(self, action, dt, **kw):
        return plain(getattr(self.g, action)(dregs=[self.v0], areg=self.t0, dt=dt, **kw))

    def test_width_letters(self):
        for dt, letter in ((adt.FP64, "e"), (adt.FP32, "w"), (adt.FP16, "h")):
            with self.subTest(dt=dt.name):
                self.assertEqual(f"vl{letter}.v v0, (t5)", self.access("load", dt))
                self.assertEqual(f"vs{letter}.v v0, (t5)", self.access("store", dt))

    def test_strides_and_broadcasts(self):
        self.assertEqual("vlse.v v0, (t5), t6",
                         self.access("load", adt.FP64, modifiers={M.GSTRIDE}, streg=self.t1))
        self.assertEqual("vssw.v v0, (t5), t6",
                         self.access("store", adt.FP32, modifiers={M.GSTRIDE}, streg=self.t1))
        self.assertEqual("vlse.v v0, (t5), zero",
                         self.access("load", adt.FP64, operand_modifiers={"adreg": {OM.BCAST}}))

    def test_indices_as_wide_as_the_data(self):
        self.assertEqual("vlxe.v v0, (t5), v1",
                         self.access("load", adt.FP64, modifiers={M.VINDEX}, vidxreg=self.v1, it=ait.INT64))
        self.assertEqual("vsuxw.v v0, (t5), v1",
                         self.access("store", adt.FP32, modifiers={M.VINDEX}, vidxreg=self.v1, it=ait.INT32))
        with self.assertRaises(ValueError):
            self.access("load", adt.FP64, modifiers={M.VINDEX}, vidxreg=self.v1, it=ait.INT32)

    def test_segments(self):
        self.assertEqual("vlseg2e.v v0, (t5)",
                         plain(self.g.load(dregs=[self.v0, self.v1], areg=self.t0, dt=adt.FP64,
                                           modifiers={M.STRUCT}, nstructs=2)))

class test_rvv071_moves_and_masks(unittest.TestCase):
    def setUp(self):
        self.g = rvv071()
        self.g.set_parameter("XTheadVector", "off")

    def test_element0_into_a_greg(self):
        text = plain(self.g.move(dregs=[self.g.vreg(1), self.g.greg(10)], dts=[adt.SINT64, adt.SINT64],
                                 modifiers=set(), operand_modifiers={"adreg": {OM.ILANE}}, adreg_lane=0))
        self.assertEqual("vext.x.v s4,v1,zero", text)

    def test_no_fp_slides(self):
        for s in self.g.move.get_signatures():
            for shape in s.operands.values():
                if OM.FILLOTHERS in (shape.modifiers or set()) and shape.dt is not None:
                    self.assertNotIn(shape.dt.name, ("DOUBLE", "SINGLE", "HALF"))

    def test_mask_count(self):
        self.assertIn("vmpopc.m", self.g.jvzero(vreg1=self.g.vreg(0), freg=self.g.freg(2, adt.FP64),
                                                 vreg2=self.g.vreg(1), greg=self.g.greg(5),
                                                 label="z", dt=adt.FP64))


class test_xtheadvector(unittest.TestCase):
    def test_on_by_default(self):
        g = rvv071()
        self.assertIn("XTheadVector", g.get_parameters())
        self.assertEqual("on", g.get_param_value("XTheadVector"))
        self.assertEqual("th.vle.v v0, (t5)", plain(g.load(dregs=[g.vreg(0)], areg=g.greg(5), dt=adt.FP64)))
        self.assertIn("th.vsetvli", g.set(dreg=g.greg(6), dt=adt.FP64, modifiers={opset_modifier.VLEN}))

    def test_scalars_and_labels_stay(self):
        g = rvv071()
        self.assertEqual("ld s4, 8(t5)", plain(g.load(dregs=[g.greg(10)], areg=g.greg(5), dt=adt.UINT64, modifiers={opmem_modifier.IOFFSET}, ioffset=8)))

    def test_off(self):
        g = rvv071()
        g.set_parameter("XTheadVector", "off")
        self.assertEqual("vle.v v0, (t5)", plain(g.load(dregs=[g.vreg(0)], areg=g.greg(5), dt=adt.FP64)))


class test_rvv071_assembles(unittest.TestCase):
    def test_forms(self):
        tool = "riscv64-linux-gnu-as"
        if not shutil.which(tool):
            self.skipTest(f"{tool} not found")
        g = rvv071()
        g.set_output_inline(False)
        v0, v1, t0, t1 = g.vreg(0), g.vreg(1), g.greg(5), g.greg(6)
        text = "".join([
            g.set(dreg=t1, dt=adt.FP64, modifiers={opset_modifier.VLEN}),
            g.load(dregs=[v0], areg=t0, dt=adt.FP64),
            g.load(dregs=[v0], areg=t0, dt=adt.FP32, modifiers={M.GSTRIDE}, streg=t1),
            g.load(dregs=[v0], areg=t0, dt=adt.FP64, modifiers={M.VINDEX}, vidxreg=v1, it=ait.INT64),
            g.store(dregs=[v0], areg=t0, dt=adt.FP32, modifiers={M.VINDEX}, vidxreg=v1, it=ait.INT32),
            g.load(dregs=[v0, v1], areg=t0, dt=adt.FP64, modifiers={M.STRUCT}, nstructs=2),
            g.move(dregs=[v1, g.greg(10)], dts=[adt.SINT64, adt.SINT64], modifiers=set(),
                   operand_modifiers={"adreg": {OM.ILANE}}, adreg_lane=0),
        ])
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "a.s")
            with open(src, "w") as f:
                f.write(text)
            r = subprocess.run([tool, "-march=rv64gc_xtheadvector", src, "-o", os.path.join(d, "a.o")],
                               capture_output=True, text=True)
        if r.returncode and "xtheadvector" in r.stderr:
            self.skipTest("the assembler has no XTheadVector")
        self.assertEqual(0, r.returncode, r.stderr + text)


if __name__ == "__main__":
    unittest.main()
