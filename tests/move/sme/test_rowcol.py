# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Tests SME row/col extracts/inserts
"""
import unittest

from asmgen.asmblocks.op import (
    opmove_modifier as mod,
    operand_modifier as opd_mod,
    make_ord_prefix as mop
)
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.aarch64.types import aarch64_greg
from asmgen.asmblocks.sve.types import sve_vreg,sve_preg
from asmgen.asmblocks.sme.types import sme_treg
from asmgen.asmblocks.sme import sme

# fine for testing
# pylint: disable-next=too-many-instance-attributes
class test_sme_opmem(unittest.TestCase):
    """
    Testsuite for SME move operations
    """
    def setUp(self):
        self.w12 = aarch64_greg(12)

        self.z0 = sve_vreg(0)
        self.z1 = sve_vreg(1)
        self.z2 = sve_vreg(2)
        self.z3 = sve_vreg(3)

        self.za0 = sme_treg(0, dt=adt.FP64)

        self.p0  = sve_preg(0)

        self.gen = sme()
        self.gen.set_output_inline(yesno=False)

    def test_sve_dispatch(self):
        """
        Tests that moves handled by the sve implementation are handled
        """

        self.assertEqual(
                "dup z0.d,z1.d[3]\n",
                self.gen.move(
                    dregs=[self.z1,self.z0],
                    dts=[adt.FP64,adt.FP64],
                    operand_modifiers={
                        'adreg': {opd_mod.ILANE},
                        'bdreg': {opd_mod.BCAST}
                        },
                    adreg_lane=3
                    ))


    def test_v_to_t_fp64(self):
        """
        Test inserting a column into a tile
        """

        self.assertEqual(
                "mova za0v.d[w12,0],p0/m,z0.d\n",
                self.gen.move(
            dregs=[self.z0,self.za0],
            dts=[adt.FP64 for _ in range(2)],
            modifiers={mod.MASK},
            amreg=self.p0,
            operand_modifiers={'bdreg':{opd_mod.COL}},
            bdreg_immcol=0,
            bdreg_colreg=self.w12))

    def test_t_to_v_fp64(self):
        """
        Test extracting a column from a tile
        """

        self.assertEqual(
                "mova z0.d,p0/m,za0v.d[w12,0]\n",
                self.gen.move(
            dregs=[self.za0,self.z0],
            dts=[adt.FP64 for _ in range(2)],
            modifiers={mod.MASK},
            amreg=self.p0,
            operand_modifiers={'adreg':{opd_mod.COL}},
            adreg_immcol=0,
            adreg_colreg=self.w12))

    def test_4rows_in_fp64(self):
        """
        Test inserting 4 rows into a tile
        """

        self.assertEqual(
                "mova za0h.d[w12,0:3],p0/m,{z0.d-z3.d}\n",
                self.gen.move(
            dregs=[self.z0,self.z1,self.z2,self.z3,
                   self.za0,self.za0,self.za0,self.za0],
            dts=[adt.FP64 for _ in range(8)],
            modifiers={mod.MASK,mod.MULTIPLE_IN,mod.MULTIPLE_OUT},
            amreg=self.p0,
            operand_modifiers={f'{mop(i+4)}dreg':{opd_mod.ROW}
                               for i in range(4)},
            nin = 4,
            nout = 4,
            edreg_immrow=0,
            fdreg_immrow=1,
            gdreg_immrow=2,
            hdreg_immrow=3,
            edreg_rowreg=self.w12,
            fdreg_rowreg=self.w12,
            gdreg_rowreg=self.w12,
            hdreg_rowreg=self.w12,
            ))
