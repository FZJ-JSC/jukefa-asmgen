# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Generator for tests of SIMD/vector ASM
"""

from asmgen.asmblocks.op import opd3_modifier as mod_d3
from asmgen.asmblocks.noarch import asm_data_type as dt, reg_tracker
from asmgen.cppgen.checkers import always_true, vec_fp64_close, vec_fp32_close
from asmgen.cppgen.expressions import identity
from asmgen.cppgen.declarations import vargen, vio_type
from asmgen.asmblocks.op.opset import opset_modifier
from asmgen.asmblocks.op.opd3 import opd3_modifier
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.op.opmem import opmem_modifier


def _vector_memory(gen, action, vreg, areg, dt):
    """
    A vector load or store of a whole vector at areg, under the ISA's
    all-true predicate where it has one (SVE's) - as the old vector
    wrappers wrote it
    """
    kw = {}
    predicate = gen.isa_setup().all_true_predicate(rt=None, dt=dt)
    if predicate is not None:
        kw = {"modifiers": {opmem_modifier.MASK}, "amreg": predicate}
    return getattr(gen, action)(dregs=[vreg], areg=areg, dt=dt, **kw)


# These will always be complex
# pylint: disable=too-many-locals

class vec_test_generator:
    """
    Generator of tests of operations on SIMD/vector registers
    """
    @classmethod
    def generate_mov_fp32_to_vreg(cls, testcase, rt : reg_tracker, vg : vargen):
        """
        Tests correctness of an ASM block that moves FP32 values from
        memory to a SIMD/vector register
        """
        test_name = "mov_fp32_to_vreg"
        imm = testcase.random_fimmediate()
        varname = vg.new_vector(cpp_type="float",
                                size="get_simd_size()/sizeof(float)",
                                fillwith=imm,
                                vt=vio_type.INPUT)


        asmblock = ""

        asmblock += testcase.gen.isaquirks(rt=rt, dt=dt.SINGLE)

        areg_idx = rt.reserve_any_reg(type_tag="greg")
        areg = testcase.gen.greg(areg_idx)
        asmblock += testcase.gen.set(dreg=areg, dt=adt.UINT64, modifiers={opset_modifier.CONST}, constant="zero")
        asmblock += testcase.gen.load(dregs=[areg], dt=adt.UINT64, modifiers={opmem_modifier.PARAM}, param=varname)

        vreg_idx = rt.reserve_any_reg(type_tag="vreg")
        vreg = testcase.gen.vreg(vreg_idx)
        asmblock += testcase.gen.set(dreg=vreg, dt=dt.SINGLE, modifiers={opset_modifier.CONST}, constant="zero")
        asmblock += _vector_memory(testcase.gen, "load", vreg, areg, dt.SINGLE)

        rt.unuse_reg(type_tag="greg", idx=areg_idx)
        rt.unuse_reg(type_tag="vreg", idx=vreg_idx)

        check_function = always_true(f"check_{test_name}")
        testcase.add_test(name=test_name, rt=rt, vg=vg,
                          asmblock=asmblock,
                          check_function_definition=check_function,
                          check=f"check_{test_name}()")

    @classmethod
    def generate_mov_fp64_to_vreg(cls, testcase, rt : reg_tracker, vg : vargen):
        """
        Tests correctness of an ASM block that moves FP64 values from
        memory to a SIMD/vector register
        """
        test_name = "mov_fp64_to_vreg"
        imm = testcase.random_fimmediate()
        varname = vg.new_vector(cpp_type="double",
                                size="get_simd_size()/sizeof(double)",
                                fillwith=imm,
                                vt=vio_type.INPUT)


        asmblock = ""
        asmblock += testcase.gen.isaquirks(rt=rt, dt=dt.DOUBLE)

        areg_idx = rt.reserve_any_reg(type_tag="greg")
        areg = testcase.gen.greg(areg_idx)
        asmblock += testcase.gen.set(dreg=areg, dt=adt.UINT64, modifiers={opset_modifier.CONST}, constant="zero")
        asmblock += testcase.gen.load(dregs=[areg], dt=adt.UINT64, modifiers={opmem_modifier.PARAM}, param=varname)

        vreg_idx = rt.reserve_any_reg(type_tag="vreg")
        vreg = testcase.gen.vreg(vreg_idx)
        asmblock += testcase.gen.set(dreg=vreg, dt=dt.DOUBLE, modifiers={opset_modifier.CONST}, constant="zero")
        asmblock += _vector_memory(testcase.gen, "load", vreg, areg, dt.DOUBLE)

        rt.unuse_reg(type_tag="greg", idx=areg_idx)
        rt.unuse_reg(type_tag="vreg", idx=vreg_idx)

        check_function = always_true(f"check_{test_name}")
        testcase.add_test(name=test_name, rt=rt, vg=vg,
                          asmblock=asmblock,
                          check_function_definition=check_function,
                          check=f"check_{test_name}()")

    @classmethod
    def generate_vec_fma_fp32(cls, testcase, rt : reg_tracker, vg : vargen):
        """
        Tests correctness of an ASM block that performs an FMA instruction
        on SIMD/vector registers filled with FP32 elements
        """
        test_name = "vec_fma_fp32"

        simd_nelements = "get_simd_size()/sizeof(float)"

        imm1 = testcase.random_fimmediate()
        avec = vg.new_vector(cpp_type="float",
                             size=simd_nelements,
                             fillwith=imm1,
                             vt=vio_type.INPUT)
        imm2 = testcase.random_fimmediate()
        bvec = vg.new_vector(cpp_type="float",
                             size=simd_nelements,
                             fillwith=imm2,
                             vt=vio_type.INPUT)
        imm3 = testcase.random_fimmediate()
        cvec = vg.new_vector(cpp_type="float",
                             size=simd_nelements,
                             fillwith=imm3,
                             vt=vio_type.INPUT)

        avreg_idx = rt.reserve_any_reg(type_tag="vreg")
        avreg = testcase.gen.vreg(avreg_idx)
        bvreg_idx = rt.reserve_any_reg(type_tag="vreg")
        bvreg = testcase.gen.vreg(bvreg_idx)
        cvreg_idx = rt.reserve_any_reg(type_tag="vreg")
        cvreg = testcase.gen.vreg(cvreg_idx)

        asmblock  = testcase.gen.isaquirks(rt=rt, dt=dt.SINGLE)

        asmblock += testcase.gen.set(dreg=avreg, dt=dt.SINGLE, modifiers={opset_modifier.CONST}, constant="zero")
        asmblock += testcase.gen.set(dreg=bvreg, dt=dt.SINGLE, modifiers={opset_modifier.CONST}, constant="zero")
        asmblock += testcase.gen.set(dreg=cvreg, dt=dt.SINGLE, modifiers={opset_modifier.CONST}, constant="zero")


        aareg_idx = rt.reserve_any_reg(type_tag="greg")
        aareg = testcase.gen.greg(aareg_idx)
        asmblock += testcase.gen.load(dregs=[aareg], dt=adt.UINT64, modifiers={opmem_modifier.PARAM}, param=avec)
        asmblock += _vector_memory(testcase.gen, "load", avreg, aareg, dt.SINGLE)
        rt.unuse_reg(type_tag="greg", idx=aareg_idx)

        bareg_idx = rt.reserve_any_reg(type_tag="greg")
        bareg = testcase.gen.greg(bareg_idx)
        asmblock += testcase.gen.load(dregs=[bareg], dt=adt.UINT64, modifiers={opmem_modifier.PARAM}, param=bvec)
        asmblock += _vector_memory(testcase.gen, "load", bvreg, bareg, dt.SINGLE)
        rt.unuse_reg(type_tag="greg", idx=bareg_idx)

        careg_idx = rt.reserve_any_reg(type_tag="greg")
        careg = testcase.gen.greg(careg_idx)
        asmblock += testcase.gen.load(dregs=[careg], dt=adt.UINT64, modifiers={opmem_modifier.PARAM}, param=cvec)
        asmblock += _vector_memory(testcase.gen, "load", cvreg, careg, dt.SINGLE)
        try:
            asmblock += testcase.gen.fma(adreg=avreg, bdreg=bvreg, cdreg=cvreg,
                                         a_dt=dt.SINGLE, b_dt=dt.SINGLE, c_dt=dt.SINGLE)
        except:
            asmblock += testcase.gen.fma(adreg=avreg, bdreg=bvreg, cdreg=cvreg,
                                         amreg=testcase.gen.mreg(0),
                                         modifiers={mod_d3.MASK},
                                         a_dt=dt.SINGLE, b_dt=dt.SINGLE, c_dt=dt.SINGLE)
        asmblock += _vector_memory(testcase.gen, "store", cvreg, careg, dt.SINGLE)
        rt.unuse_reg(type_tag="greg", idx=careg_idx)

        rt.unuse_reg(type_tag="vreg", idx=avreg_idx)
        rt.unuse_reg(type_tag="vreg", idx=bvreg_idx)
        rt.unuse_reg(type_tag="vreg", idx=cvreg_idx)

        extra_prepare  = f"std::vector<float> result({simd_nelements});\n"
        # adding up in python will actually result in failures
        extra_prepare += f"std::fill(result.begin(), result.end(), {imm3}+{imm1}*{imm2});\n"

        check_function = vec_fp32_close(f"check_{test_name}", identity)
        testcase.add_test(name=test_name, rt=rt, vg=vg,
                          asmblock=asmblock,
                          check_function_definition=check_function,
                          check=f"check_{test_name}({cvec}, result)",
                          extra_prepare=extra_prepare)

    @classmethod
    def generate_vec_fma_fp64(cls, testcase, rt : reg_tracker, vg : vargen):
        """
        Tests correctness of an ASM block that performs an FMA instruction
        on SIMD/vector registers filled with FP64 elements
        """
        test_name = "vec_fma_fp64"

        simd_nelements = "get_simd_size()/sizeof(double)"

        imm1 = testcase.random_fimmediate()
        avec = vg.new_vector(cpp_type="double",
                             size=simd_nelements,
                             fillwith=imm1,
                             vt=vio_type.INPUT)
        imm2 = testcase.random_fimmediate()
        bvec = vg.new_vector(cpp_type="double",
                             size=simd_nelements,
                             fillwith=imm2,
                             vt=vio_type.INPUT)
        imm3 = testcase.random_fimmediate()
        cvec = vg.new_vector(cpp_type="double",
                             size=simd_nelements,
                             fillwith=imm3,
                             vt=vio_type.INPUT)

        avreg_idx = rt.reserve_any_reg(type_tag="vreg")
        avreg = testcase.gen.vreg(avreg_idx)
        bvreg_idx = rt.reserve_any_reg(type_tag="vreg")
        bvreg = testcase.gen.vreg(bvreg_idx)
        cvreg_idx = rt.reserve_any_reg(type_tag="vreg")
        cvreg = testcase.gen.vreg(cvreg_idx)

        asmblock  = testcase.gen.isaquirks(rt=rt, dt=dt.DOUBLE)

        asmblock += testcase.gen.set(dreg=avreg, dt=dt.DOUBLE, modifiers={opset_modifier.CONST}, constant="zero")
        asmblock += testcase.gen.set(dreg=bvreg, dt=dt.DOUBLE, modifiers={opset_modifier.CONST}, constant="zero")
        asmblock += testcase.gen.set(dreg=cvreg, dt=dt.DOUBLE, modifiers={opset_modifier.CONST}, constant="zero")


        aareg_idx = rt.reserve_any_reg(type_tag="greg")
        aareg = testcase.gen.greg(aareg_idx)
        asmblock += testcase.gen.load(dregs=[aareg], dt=adt.UINT64, modifiers={opmem_modifier.PARAM}, param=avec)
        asmblock += _vector_memory(testcase.gen, "load", avreg, aareg, dt.DOUBLE)
        rt.unuse_reg(type_tag="greg", idx=aareg_idx)

        bareg_idx = rt.reserve_any_reg(type_tag="greg")
        bareg = testcase.gen.greg(bareg_idx)
        asmblock += testcase.gen.load(dregs=[bareg], dt=adt.UINT64, modifiers={opmem_modifier.PARAM}, param=bvec)
        asmblock += _vector_memory(testcase.gen, "load", bvreg, bareg, dt.DOUBLE)
        rt.unuse_reg(type_tag="greg", idx=bareg_idx)

        careg_idx = rt.reserve_any_reg(type_tag="greg")
        careg = testcase.gen.greg(careg_idx)
        asmblock += testcase.gen.load(dregs=[careg], dt=adt.UINT64, modifiers={opmem_modifier.PARAM}, param=cvec)
        asmblock += _vector_memory(testcase.gen, "load", cvreg, careg, dt.DOUBLE)
        try:
            asmblock += testcase.gen.fma(adreg=avreg, bdreg=bvreg, cdreg=cvreg,
                                         a_dt=dt.DOUBLE, b_dt=dt.DOUBLE, c_dt=dt.DOUBLE)
        except:
            asmblock += testcase.gen.fma(adreg=avreg, bdreg=bvreg, cdreg=cvreg,
                                         amreg=testcase.gen.mreg(0),
                                         modifiers={mod_d3.MASK},
                                         a_dt=dt.DOUBLE, b_dt=dt.DOUBLE, c_dt=dt.DOUBLE)
        asmblock += _vector_memory(testcase.gen, "store", cvreg, careg, dt.DOUBLE)
        rt.unuse_reg(type_tag="greg", idx=careg_idx)

        rt.unuse_reg(type_tag="vreg", idx=avreg_idx)
        rt.unuse_reg(type_tag="vreg", idx=bvreg_idx)
        rt.unuse_reg(type_tag="vreg", idx=cvreg_idx)

        extra_prepare  = f"std::vector<double> result({simd_nelements});\n"
        # adding up in python will actually result in failures
        extra_prepare += f"std::fill(result.begin(), result.end(), {imm3}+{imm1}*{imm2});\n"

        check_function = vec_fp64_close(f"check_{test_name}", identity)
        testcase.add_test(name=test_name, rt=rt, vg=vg,
                          asmblock=asmblock,
                          check_function_definition=check_function,
                          check=f"check_{test_name}({cvec}, result)",
                          extra_prepare=extra_prepare)

    @classmethod
    def generate_vec_mul_fp32(cls, testcase, rt : reg_tracker, vg : vargen):
        """
        Tests correctness of an ASM block that performs an FMUL instruction
        on SIMD/vector registers filled with FP32 elements
        """
        test_name = "vec_mul_fp32"

        simd_nelements = "get_simd_size()/sizeof(float)"

        imm1 = testcase.random_fimmediate()
        avec = vg.new_vector(cpp_type="float",
                             size=simd_nelements,
                             fillwith=imm1,
                             vt=vio_type.INPUT)
        imm2 = testcase.random_fimmediate()
        bvec = vg.new_vector(cpp_type="float",
                             size=simd_nelements,
                             fillwith=imm2,
                             vt=vio_type.INPUT)

        avreg_idx = rt.reserve_any_reg(type_tag="vreg")
        avreg = testcase.gen.vreg(avreg_idx)
        bvreg_idx = rt.reserve_any_reg(type_tag="vreg")
        bvreg = testcase.gen.vreg(bvreg_idx)

        asmblock  = testcase.gen.isaquirks(rt=rt, dt=dt.SINGLE)

        asmblock += testcase.gen.set(dreg=avreg, dt=dt.SINGLE, modifiers={opset_modifier.CONST}, constant="zero")
        asmblock += testcase.gen.set(dreg=bvreg, dt=dt.SINGLE, modifiers={opset_modifier.CONST}, constant="zero")

        aareg_idx = rt.reserve_any_reg(type_tag="greg")
        aareg = testcase.gen.greg(aareg_idx)
        asmblock += testcase.gen.load(dregs=[aareg], dt=adt.UINT64, modifiers={opmem_modifier.PARAM}, param=avec)
        asmblock += _vector_memory(testcase.gen, "load", avreg, aareg, dt.SINGLE)

        bareg_idx = rt.reserve_any_reg(type_tag="greg")
        bareg = testcase.gen.greg(bareg_idx)
        asmblock += testcase.gen.load(dregs=[bareg], dt=adt.UINT64, modifiers={opmem_modifier.PARAM}, param=bvec)
        asmblock += _vector_memory(testcase.gen, "load", bvreg, bareg, dt.SINGLE)
        rt.unuse_reg(type_tag="greg", idx=bareg_idx)
        try:
            asmblock += testcase.gen.mul(adreg=avreg, bdreg=bvreg, cdreg=avreg,
                                         a_dt=dt.SINGLE, b_dt=dt.SINGLE, c_dt=dt.SINGLE)
        except:
            asmblock += testcase.gen.mul(adreg=avreg, bdreg=bvreg, cdreg=avreg,
                                          amreg=testcase.gen.mreg(0),
                                          modifiers={mod_d3.MASK},
                                          a_dt=dt.SINGLE, b_dt=dt.SINGLE, c_dt=dt.SINGLE)
        asmblock += _vector_memory(testcase.gen, "store", avreg, aareg, dt.SINGLE)
        rt.unuse_reg(type_tag="greg", idx=aareg_idx)

        rt.unuse_reg(type_tag="vreg", idx=avreg_idx)
        rt.unuse_reg(type_tag="vreg", idx=bvreg_idx)

        extra_prepare  = f"std::vector<float> result({simd_nelements});\n"
        # adding up in python will actually result in failures
        extra_prepare += f"std::fill(result.begin(), result.end(), {imm1}*{imm2});\n"

        check_function = vec_fp32_close(f"check_{test_name}", identity)
        testcase.add_test(name=test_name, rt=rt, vg=vg,
                          asmblock=asmblock,
                          check_function_definition=check_function,
                          check=f"check_{test_name}({avec}, result)",
                          extra_prepare=extra_prepare)

    @classmethod
    def generate_vec_mul_fp64(cls, testcase, rt : reg_tracker, vg : vargen):
        """
        Tests correctness of an ASM block that performs an FMUL instruction
        on SIMD/vector registers filled with FP64 elements
        """
        test_name = "vec_mul_fp64"

        simd_nelements = "get_simd_size()/sizeof(double)"

        imm1 = testcase.random_fimmediate()
        avec = vg.new_vector(cpp_type="double",
                             size=simd_nelements,
                             fillwith=imm1,
                             vt=vio_type.INPUT)
        imm2 = testcase.random_fimmediate()
        bvec = vg.new_vector(cpp_type="double",
                             size=simd_nelements,
                             fillwith=imm2,
                             vt=vio_type.INPUT)

        avreg_idx = rt.reserve_any_reg(type_tag="vreg")
        avreg = testcase.gen.vreg(avreg_idx)
        bvreg_idx = rt.reserve_any_reg(type_tag="vreg")
        bvreg = testcase.gen.vreg(bvreg_idx)

        asmblock  = testcase.gen.isaquirks(rt=rt, dt=dt.DOUBLE)

        asmblock += testcase.gen.set(dreg=avreg, dt=dt.DOUBLE, modifiers={opset_modifier.CONST}, constant="zero")
        asmblock += testcase.gen.set(dreg=bvreg, dt=dt.DOUBLE, modifiers={opset_modifier.CONST}, constant="zero")

        aareg_idx = rt.reserve_any_reg(type_tag="greg")
        aareg = testcase.gen.greg(aareg_idx)
        asmblock += testcase.gen.load(dregs=[aareg], dt=adt.UINT64, modifiers={opmem_modifier.PARAM}, param=avec)
        asmblock += _vector_memory(testcase.gen, "load", avreg, aareg, dt.DOUBLE)

        bareg_idx = rt.reserve_any_reg(type_tag="greg")
        bareg = testcase.gen.greg(bareg_idx)
        asmblock += testcase.gen.load(dregs=[bareg], dt=adt.UINT64, modifiers={opmem_modifier.PARAM}, param=bvec)
        asmblock += _vector_memory(testcase.gen, "load", bvreg, bareg, dt.DOUBLE)
        rt.unuse_reg(type_tag="greg", idx=bareg_idx)

        try:
            asmblock += testcase.gen.mul(adreg=avreg, bdreg=bvreg, cdreg=avreg,
                                         a_dt=dt.DOUBLE, b_dt=dt.DOUBLE, c_dt=dt.DOUBLE)
        except:
            asmblock += testcase.gen.mul(adreg=avreg, bdreg=bvreg, cdreg=avreg,
                                          amreg=testcase.gen.mreg(0),
                                          modifiers={mod_d3.MASK},
                                          a_dt=dt.DOUBLE, b_dt=dt.DOUBLE, c_dt=dt.DOUBLE)
        asmblock += _vector_memory(testcase.gen, "store", avreg, aareg, dt.DOUBLE)
        rt.unuse_reg(type_tag="greg", idx=aareg_idx)

        rt.unuse_reg(type_tag="vreg", idx=avreg_idx)
        rt.unuse_reg(type_tag="vreg", idx=bvreg_idx)

        extra_prepare  = f"std::vector<double> result({simd_nelements});\n"
        # adding up in python will actually result in failures
        extra_prepare += f"std::fill(result.begin(), result.end(), {imm1}*{imm2});\n"

        check_function = vec_fp64_close(f"check_{test_name}", identity)
        testcase.add_test(name=test_name, rt=rt, vg=vg,
                          asmblock=asmblock,
                          check_function_definition=check_function,
                          check=f"check_{test_name}({avec}, result)",
                          extra_prepare=extra_prepare)
