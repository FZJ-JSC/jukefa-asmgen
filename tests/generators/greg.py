# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Generator for tests of GP register-related ASM blocks
"""
from asmgen.registers import reg_tracker
from asmgen.cppgen.checkers import bi_u64_eq
from asmgen.cppgen.declarations import vargen
from asmgen.cppgen.expressions import identity

from ..testcase import testcase
from asmgen.asmblocks.op.opset import opset_modifier
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.op.opmem import opmem_modifier

class greg_test_generator:
    """
    Generator of tests of operations on GP register
    """
    ###################################
    # Test code generators start here #
    ###################################

    @classmethod
    def generate_mov_imm_to_greg(cls, tcase : testcase,
                                 rt : reg_tracker, vg : vargen):
        """
        Tests correctness of an ASM block that moves an immediate into a GP register
        """
        test_name = "mov_imm_to_greg"

        varname = vg.new_var(cpp_type="std::uint64_t")
        reg_idx = rt.reserve_any_reg(type_tag="greg")
        reg = tcase.gen.greg(reg_idx=reg_idx)
        imm = tcase.random_immediate()

        asmblock  = tcase.gen.set(dreg=reg, dt=adt.UINT64, modifiers={opset_modifier.CONST}, constant="zero")
        asmblock += tcase.gen.set(dreg=reg, dt=adt.UINT64, modifiers={opset_modifier.CONST}, value=imm)
        asmblock += tcase.gen.store(dregs=[reg], dt=adt.UINT64, modifiers={opmem_modifier.PARAM}, param=varname)

        rt.unuse_reg(type_tag="greg", idx=reg_idx)
        check_function = bi_u64_eq(f"check_{test_name}",identity)
        tcase.add_test(name=test_name, rt=rt, vg=vg,
                       asmblock=asmblock,
                       check_function_definition=check_function,
                       check=f"check_{test_name}({varname}, {imm})")

    @classmethod
    def generate_add_imm_to_greg(cls, tcase : testcase, rt : reg_tracker, vg : vargen):
        """
        Tests correctness of an ASM block that adds an immediate to a GP register
        """
        test_name = "add_imm_to_greg"

        varname = vg.new_var(cpp_type="std::uint64_t")
        reg_idx = rt.reserve_any_reg(type_tag="greg")
        reg = tcase.gen.greg(reg_idx)
        imm = tcase.random_immediate()
        asmblock  = tcase.gen.set(dreg=reg, dt=adt.UINT64, modifiers={opset_modifier.CONST}, constant="zero")
        asmblock += tcase.gen.add(adreg=reg, bimm=imm, cdreg=reg, a_dt=adt.UINT64, c_dt=adt.UINT64)
        asmblock += tcase.gen.store(dregs=[reg], dt=adt.UINT64, modifiers={opmem_modifier.PARAM}, param=varname)
        rt.unuse_reg(type_tag="greg", idx=reg_idx)
        check_function = bi_u64_eq(f"check_{test_name}",identity)
        tcase.add_test(name=test_name, rt=rt, vg=vg,
                       asmblock=asmblock,
                       check_function_definition=check_function,
                       check=f"check_{test_name}({varname}, {imm})")

    @classmethod
    def generate_add_imm_to_greg_2_times(cls, tcase : testcase, rt : reg_tracker, vg : vargen):
        """
        Tests correctness of an ASM block that adds an immediate to a GP register 2 times
        """
        test_name = "add_imm_to_greg_2_times"

        varname = vg.new_var(cpp_type="std::uint64_t")
        reg_idx = rt.reserve_any_reg(type_tag="greg")
        reg = tcase.gen.greg(reg_idx)
        imm = tcase.random_immediate()
        asmblock  = tcase.gen.set(dreg=reg, dt=adt.UINT64, modifiers={opset_modifier.CONST}, constant="zero")
        asmblock += tcase.gen.add(adreg=reg, bimm=imm, cdreg=reg, a_dt=adt.UINT64, c_dt=adt.UINT64)
        asmblock += tcase.gen.add(adreg=reg, bimm=imm, cdreg=reg, a_dt=adt.UINT64, c_dt=adt.UINT64)
        asmblock += tcase.gen.store(dregs=[reg], dt=adt.UINT64, modifiers={opmem_modifier.PARAM}, param=varname)
        rt.unuse_reg(type_tag="greg", idx=reg_idx)
        check_function = bi_u64_eq(f"check_{test_name}",identity)
        tcase.add_test(name=test_name, rt=rt, vg=vg,
                      asmblock=asmblock,
                      check_function_definition=check_function,
                      check=f"check_{test_name}({varname}, {imm+imm})")

    @classmethod
    def generate_add_greg_to_greg(cls, tcase : testcase, rt : reg_tracker, vg : vargen):
        """
        Tests correctness of an ASM block that adds the value of a GP register
        to another GP register
        """
        test_name = "add_greg_to_greg"

        varname = vg.new_var(cpp_type="std::uint64_t")
        dst_idx = rt.reserve_any_reg(type_tag="greg")
        reg1_idx = rt.reserve_any_reg(type_tag="greg")
        reg2_idx = rt.reserve_any_reg(type_tag="greg")

        dst = tcase.gen.greg(dst_idx)
        reg1 = tcase.gen.greg(reg1_idx)
        reg2 = tcase.gen.greg(reg2_idx)

        imm = tcase.random_immediate()

        asmblock  = tcase.gen.set(dreg=dst, dt=adt.UINT64, modifiers={opset_modifier.CONST}, constant="zero")
        asmblock += tcase.gen.set(dreg=reg1, dt=adt.UINT64, modifiers={opset_modifier.CONST}, constant="zero")
        asmblock += tcase.gen.set(dreg=reg2, dt=adt.UINT64, modifiers={opset_modifier.CONST}, constant="zero")
        asmblock += tcase.gen.add(adreg=reg1, bimm=imm, cdreg=reg1, a_dt=adt.UINT64, c_dt=adt.UINT64)
        asmblock += tcase.gen.add(adreg=reg2, bimm=imm, cdreg=reg2, a_dt=adt.UINT64, c_dt=adt.UINT64)
        asmblock += tcase.gen.add(adreg=reg1, bdreg=reg2, cdreg=dst, a_dt=adt.UINT64, b_dt=adt.UINT64, c_dt=adt.UINT64)
        asmblock += tcase.gen.store(dregs=[dst], dt=adt.UINT64, modifiers={opmem_modifier.PARAM}, param=varname)

        rt.unuse_reg(type_tag="greg", idx=dst_idx)
        rt.unuse_reg(type_tag="greg", idx=reg1_idx)
        rt.unuse_reg(type_tag="greg", idx=reg2_idx)

        check_function = bi_u64_eq(f"check_{test_name}",identity)
        tcase.add_test(name=test_name, rt=rt, vg=vg,
                      asmblock=asmblock,
                      check_function_definition=check_function,
                      check=f"check_{test_name}({varname}, {imm+imm})")
