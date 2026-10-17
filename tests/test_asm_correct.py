# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Test the asmgen implementations for different ISAs on whether they generate
correct code
"""

import logging
import os
import platform
import random
import shutil
import unittest

from subprocess import Popen, PIPE
from typing import Union

from parameterized import parameterized, parameterized_class

from asmgen.registers import reg_tracker
from asmgen.asmblocks.noarch import asmgen
from asmgen.asmblocks.avx import fma128,fma256,avx512
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.rvv import rvv
from asmgen.cppgen.writers import write_test_asmblock_func
from asmgen.cppgen.writers import write_test_func_declaration
from asmgen.cppgen.declarations import vargen,vio_type
from asmgen.compilation.tools import compiler
from asmgen.compilation.compiler_presets import cross_archs,cross_cxx_flags
from asmgen.emulation import get_emulator

from .testcase import testcase
from .asm_test_generator import asm_test_generator


#NOTE: rvv071 will only work with special 0.7.1 EPI llvm, which will not work with RVV 1.0
#      So let's remove testing it
gens = [fma128(), fma256(), avx512(), neon(), sve(), rvv()]
gen_dicts : list[dict[str,Union[str,asmgen]]] = [{
    "name": gen.__class__.__name__,
    "gen": gen
    } for gen in gens]
cxx_dicts = [{
    "cxx_name" : name,
    } for name in ["g++", "clang++", "armclang++"]]
# NOTE: this needs python 3.9 or higher
parameters = [d1 | d2 for d1 in gen_dicts for d2 in cxx_dicts]

@parameterized_class(parameters)
class asm_correctness_test(unittest.TestCase,testcase):
    """
    Runs code correctness tests on ASM generators. To add a test, add it to one of the
    generators in tests/generators or create a new one and make asm_test_generator in
    asm_test_generator.py inherit from it
    """
    test_root = "test_run/asm_correct/"

    @classmethod
    def clang_format_if_exists(cls, source : str) -> str:
        """
        Prettify the source with clang-format if the binary can be found
        """
        clang_format_exe = shutil.which("clang-format")
        if None is not clang_format_exe:
            cmd = f"{clang_format_exe}"
            with Popen([cmd,"--style=llvm"],
                stdin=PIPE, stdout=PIPE, stderr=PIPE) as p:
                pretty_source = p.communicate(input=source.encode())[0].decode()
                if 0 == p.returncode:
                    return pretty_source
        return source

    @classmethod
    def setUpClass(cls):
        # only used for existence check
        compiler_exe = shutil.which(cls.cxx_name)
        if None is compiler_exe:
            raise unittest.SkipTest(f"Compiler {cls.cxx_name} not found, skipping")

        cls.testlib_name = f"libtest_asm_correct_{cls.name}.so"
        cls.testlib_source  = f"// uarch_bench; Generator: {cls.name}\n"
        cls.testlib_source += "// generated C++/inline ASM code for ASM correctness tests\n"
        cls.header_source  = f"// uarch_bench; Generator: {cls.name}\n"
        cls.header_source += "// generated C++ header for ASM correctness tests\n"

        log = logging.getLogger("ASMCORTEST")

        cls.testlib_source += "#include <cstddef>\n"
        cls.testlib_source += "#include <cstdlib>\n"
        cls.testlib_source += "#include <cmath>\n"
        cls.testlib_source += "#include <limits>\n"
        # Add simd_size function so it's available for tests
        cls.testlib_source += cls.gen.c_simd_size_function

        # Generating test sources
        vg = vargen()
        reg_init_list = [
                ("greg", cls.gen.max_gregs()),
                ("vreg", cls.gen.max_vregs()),
                ("freg", cls.gen.max_fregs())
                ]
        rt = reg_tracker(reg_type_init_list=reg_init_list)

        # Run all generators
        generators = [getattr(asm_test_generator,name) for\
            name in dir(asm_test_generator) if name.startswith("generate_")]
        for generator in generators:
            generator(cls,rt,vg)

        cls.testlib_source = cls.clang_format_if_exists(cls.testlib_source)
        cls.header_source = cls.clang_format_if_exists(cls.header_source)

        if not os.path.exists(cls.test_root):
            os.makedirs(cls.test_root)

        # save sources for later inspection
        source_path = os.path.join(cls.test_root, f"testlib_source_{cls.name}.cpp")
        header_path = os.path.join(cls.test_root, f"testlib_header_{cls.name}.hpp")

        with open(source_path, mode="w", encoding="utf-8") as source_file:
            source_file.write(cls.testlib_source)

        with open(header_path, mode="w", encoding="utf-8") as header_file:
            header_file.write(cls.header_source)

        cross_arch = cross_archs[cls.name]

        cls.compilation_failed = False
        if platform.machine() == cross_arch:
            cxx = compiler(cls.cxx_name,cls.name)
            result = cxx.compile_lib(source=cls.testlib_source,
                                     output_filename=os.path.join(
                                         cls.test_root, cls.testlib_name))
            cls.compilation_failed = 0 == result
        else:
            log.debug("code arch: %s, platform arch: %s",
                cross_arch,platform.machine())
            # Look up if cxx flags exist for compiler
            if not cls.cxx_name in cross_cxx_flags[cross_arch]:
                log.debug(("skipping testlib compilation on generator %s"
                           " because it's not supported on host and can't cross-compile"
                           "with %s"), cls.name, cls.cxx_name)
                return
            log.debug("%s not supported on host, will cross compile and emulate",cls.name)
            cxx = compiler(cls.cxx_name,cls.name)
            result = cxx.compile_lib(source=cls.testlib_source,
                                     output_filename=os.path.join(
                                         cls.test_root, cls.testlib_name),
                                     cross_compile=cross_arch)
            cls.compilation_failed = 0 == result

    def setUp(self):
        if self.compilation_failed:
            self.fail(f"Compilation failed with {self.cxx_name}!")

    @staticmethod
    def random_immediate() -> int:
        """
        Generate a random immediate/literal
        """
        return random.randint(0,2**8)

    @staticmethod
    def random_fimmediate():
        """
        Generate a random FP immediate/literal
        """
        return random.uniform(0.0,1.0)

    @classmethod
    def add_test(cls, *, name : str,
                 rt : reg_tracker,
                 vg : vargen,
                 asmblock : str,
                 check_function_definition : str,
                 check : str,
                 extra_prepare : str =""):
        """
        Adds a correctness test

        :param name: Name of the test
        :param rt: register tracker set up for the generator
        :param vg: C++ variable generator
        :param asmblock: String containing the inline ASM block to test
        :param check_function_definition: String containing the C++ definition of 
            the function that tests the outcome of the test
        :param check: String containing the C++ source of the check
        :param extra_prepare: Additional C++ source required to prepare the test
        """

        header = vg.get_includes()
        preparation = vg.get_declarations()
        preparation += "\n"
        preparation += extra_prepare
        clobbered_gregs = [cls.gen.greg(i) for i in rt.get_clobbered_regs(type_tag="greg")]
        clobbered_vregs = [cls.gen.vreg(i) for i in rt.get_clobbered_regs(type_tag="vreg")]

        inputs : list[tuple[str,str,str]] = \
                [(varname,"m",init) for varname,init,vt in vg.get_variables() \
                if vt == vio_type.INPUT]
        outputs : list[tuple[str,str,str]] = \
                [(varname,"=m",init) for varname,init,vt in vg.get_variables() \
                if vt == vio_type.OUTPUT]
        asmblock += cls.gen.operands(inputs, outputs,
                                 clobbered_gregs+clobbered_vregs+["memory"])
        rt.reset()
        vg.reset_variables()

        tparams = {
            "function_name" : name,
            "function_params" : "",
            "header" : header,
            "prepare" : preparation,
            "checkfun_definition": check_function_definition,
            "check" : check,
            "analyze" : ""
        }
        cls.testlib_source += write_test_asmblock_func(name, asmblock, tparams)
        cls.header_source += write_test_func_declaration(name, tparams)

    @staticmethod
    def wrap_test_call(function_name):
        """
        Returns the C++ source of an executable that will just call the passed function
        and return it's result

        :param function_name: Name of the function to call
        :return: C++ source of a standalone test for this function
        """
        return f"""
        int main()
        {{
            return {function_name}();
        }}
        """


    #######################
    # Test execution here #
    #######################

#    @parameterized.expand([
#        ['add_imm_to_greg'],
#        ['add_imm_to_greg_2_times'],
#        ['add_greg_to_greg'],
#        ['mov_imm_to_greg'],
#        ['mov_fp32_to_vreg'],
#        ['mov_fp64_to_vreg'],
#        ['fp32_fma'],
#        ['fp64_fma'],
#        ])
    @parameterized.expand([
        [name.replace("generate_","")] for name in dir(asm_test_generator)\
            if name.startswith("generate_")])
    def test_asm(self, test_name):
        """
        Compiles an executable for the specified test and executes it
        Fails if the compilation/execution of the test fails or the
        test doesn't return 0

        :param test_name: name of the test/function
        """
        log = logging.getLogger("ASMCORTEST")
        source = self.header_source+"\n"+self.wrap_test_call(test_name)
        cxx = compiler(self.cxx_name,self.name)
        exe : str = ""
        exe_name = f"test_{test_name}_{self.name}"

        log.debug("testing %s with %s",test_name,self.name)

        cross_arch = cross_archs[self.name]
        cmd = []

        if platform.machine() == cross_arch:
            cxx.compile_exe(source=source,
                            output_filename=os.path.join(self.test_root,exe_name),
                            libs=[os.path.join(self.test_root,self.testlib_name)])

            exe = shutil.which(exe_name,path=self.test_root)
            if None is exe:
                self.fail(f"Compilation failed with generator {self.name} for {test_name}")
            cmd = [exe]
        else:
            if not self.cxx_name in cross_cxx_flags[cross_arch]:
                self.skipTest(f"Skipping: Can't cross-compile for {self.name} with {self.cxx_name}")
            cxx.compile_exe(source=source,
                            output_filename=os.path.join(self.test_root,exe_name),
                            libs=[os.path.join(self.test_root,self.testlib_name)],
                            cross_compile=cross_arch)

            exe = shutil.which(exe_name,path=self.test_root)
            if None is exe:
                self.fail(f"Compilation failed with generator {self.name} for {test_name}")
        if not self.gen.supported_on_host():
            emulator = get_emulator(platform.machine(), self.name)
            emulator.run(exe)
        else:
            log.debug("Running test %s with: %s", test_name, ' '.join(cmd))
            output = ""
            errout = ""
            with Popen(cmd,stdin=PIPE,stdout=PIPE,stderr=PIPE) as process:
                process_out = process.communicate()
                output = process_out[0].decode()
                errout = process_out[1].decode()
                self.assertEqual(0, process.returncode)
            log.debug("Execution stdout: %s", output)
            log.debug("Execution stderr: %s", errout)


def main():
    """
    Main function (setting log levels for debugging tests)
    """
    # useful for debugging tests
    logging.basicConfig(level=logging.DEBUG, format='%(name)s %(levelname)s %(message)s')
    logging.getLogger("ASMCORTEST").setLevel(logging.DEBUG)
    logging.getLogger("COMPILATION").setLevel(logging.DEBUG)
    unittest.main()

if "__main__" == __name__:
    main()
