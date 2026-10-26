
# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Tests SVE lane and scalar moves/broadcasts
"""
import unittest

from asmgen.asmblocks.op import (
    opmove_modifier as mod,
    operand_modifier as opd_mod,
)
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.aarch64.types import aarch64_greg,aarch64_freg
from asmgen.asmblocks.sve.types import sve_vreg,sve_preg
from asmgen.asmblocks.sve import sve

# fine for testing
# pylint: disable-next=too-many-instance-attributes
class test_sve_move(unittest.TestCase):
    """
    Testsuite for SVE move operations
    """
    def setUp(self):

        self.z0 = sve_vreg(0)
        self.z1 = sve_vreg(1)
        self.z2 = sve_vreg(2)
        self.z3 = sve_vreg(3)

        self.d4 = aarch64_freg(4,dt=adt.FP64)
        self.x0 = aarch64_greg(0)

        self.p3  = sve_preg(3)

        self.gen = sve()
        self.gen.set_output_inline(yesno=False)


    def test_bcast_fp(self):
        """
        Test broadcasting from an FP register into a vector register
        """

        self.assertEqual(
                "mov z0.d,d4\n",
                self.gen.move(
                    dregs=[self.d4,self.z0],
                    dts=[adt.FP64,adt.FP64],
                    operand_modifiers={
                        'bdreg': {opd_mod.BCAST}
                        }
                    ))

    def test_bcast_fp_masked(self):
        """
        Test broadcasting from an FP register into a vector register with a predicate
        """

        self.assertEqual(
                "mov z0.d,p3/m,d4\n",
                self.gen.move(
                    dregs=[self.d4,self.z0],
                    dts=[adt.FP64,adt.FP64],
                    modifiers={mod.MASK},
                    operand_modifiers={
                        'bdreg': {opd_mod.BCAST}
                        },
                    amreg=self.p3
                    ))

    def test_bcast_lane(self):
        """
        Test broadcasting from a lane of one vector register into another vector register
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

    def test_bcast_lane_masked(self):
        """
        Test broadcasting from a lane of one vector register into another vector register 
        with a predicate
        """

        self.assertEqual(
                "dup z0.d,p3/m,z1.d[3]\n",
                self.gen.move(
                    dregs=[self.z1,self.z0],
                    dts=[adt.FP64,adt.FP64],
                    modifiers={mod.MASK},
                    operand_modifiers={
                        'adreg': {opd_mod.ILANE},
                        'bdreg': {opd_mod.BCAST}
                        },
                    adreg_lane=3,
                    amreg=self.p3
                    ))

    def test_move_vec_to_vec(self):
        """
        Test moving one vector register to another vector register
        """
        self.assertEqual(
                "mov z0.d,z1.d\n",
                self.gen.move(
                    dregs=[self.z1,self.z0],
                    dts=[adt.FP64,adt.FP64],
                    ))

    def test_move_vec_to_vec_masked(self):
        """
        test moving one vector register to another vector register with a predicate
        """
        self.assertEqual(
                "mov z0.d,p3/m,z1.d\n",
                self.gen.move(
                    dregs=[self.z1,self.z0],
                    dts=[adt.FP64,adt.FP64],
                    modifiers={mod.MASK},
                    amreg=self.p3
                    ))
