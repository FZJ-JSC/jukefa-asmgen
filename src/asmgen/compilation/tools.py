# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Compilation tools
"""

from subprocess import Popen, PIPE
import logging

from .compiler_presets import (
    default_flags,
    output_flag,
    stdin_flags,
    lib_flags,
    arch_flags,
    cross_cxx_flags,
    cross_lib_flags
)

def baseline_flags(gen, executable : str) -> list[str]:
    """
    A generator's compiler flags at its baseline - no optional extensions,
    which older compilers may not know - its features as they were after
    """
    features = gen.features
    gen.set_features([])
    try:
        return gen.compiler_flags(executable)
    finally:
        gen.set_features(sorted(gen.FEATURES) if features is None else sorted(features))


class compiler:
    """
    Simple wrapper around a compiler for directly compiling 
    sources into libraries and executables
    """
    def __init__(self, executable, arch, gen=None):
        """
        :param gen: the generator whose output it compiles: its
            compiler_flags at its baseline (no optional extensions) - the
            compiler_presets table where there's none
        """
        self.executable = executable
        self.unsupported = None
        if self.executable in default_flags:
            self.flags = default_flags[self.executable]
        if self.executable in output_flag:
            self.oflag = output_flag[self.executable]
        if self.executable in lib_flags:
            self.lib_flags = lib_flags[self.executable]
        if self.executable in stdin_flags:
            self.stdin_flags = stdin_flags[self.executable]
        if gen is not None:
            try:
                self.arch_flags = baseline_flags(gen, self.executable)
            except NotImplementedError as err:
                self.unsupported = str(err)
                self.arch_flags = []
        elif self.executable in arch_flags:
            self.arch_flags = arch_flags[self.executable][arch]

    def run_compilation(self, source : str, output_filename : str, cmd : list[str]):
        """
        Run the actual compilation
        """
        log = logging.getLogger("COMPILATION")
        log.debug("Compiling c++ code for %s using:", output_filename)
        log.debug("%s", ' '.join(cmd))
        log.debug("...")
        output = ""
        errout = ""
        rcode = 1
        with Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE) as process:
            process_out = process.communicate(input=source.encode())
            output = process_out[0].decode()
            errout = process_out[1].decode()
            rcode = process.returncode
        log.debug("Compilation stdout: %s", output)
        if 0 != rcode:
            log.error("%s", ' '.join(cmd))
            log.error("Compilation stderr: %s", errout)
        else:
            log.debug("Compilation stderr: %s", errout)

        return 0 == rcode

    # read-only-access, therefore ok
    # pylint: disable=dangerous-default-value
    def compile_lib(self, *, source : str, output_filename : str,
                    cross_compile : str ="native", extraflags : list[str] = []):
        """
        Compile a source into a library

        :param source: string containing the source code
        :type source: str
        :param output_filename: resulting library file name
        :type output_filename: str
        :param cross_compile: architecture to compile for or "native"
        :type cross_compiler: str
        :param extraflags: Additional flags to pass to the compiler
        :type extraflags: list[str]
        :return: True if the compilation succeeds, otherwise False
        :rtype: bool
        """
        if self.unsupported:
            raise NotImplementedError(f"{self.executable}: {self.unsupported}")
        cross_flags = []
        if "native" != cross_compile:
            cross_flags = cross_cxx_flags[cross_compile][self.executable] +\
                          cross_lib_flags[cross_compile][self.executable]
        cmd = [self.executable] +\
               self.flags +\
               self.arch_flags +\
               self.lib_flags +\
               cross_flags +\
               extraflags +\
               [self.oflag,output_filename] +\
               self.stdin_flags

        return self.run_compilation(source=source, output_filename=output_filename, cmd=cmd)


    # read-only-access, therefore ok
    # pylint: disable=dangerous-default-value
    def compile_exe(self, *, source : str, output_filename : str, libs : list[str],
                cross_compile : str ="native", extraflags : list[str] = []):
        """
        Compile a source into an executable

        :param source: string containing the source code
        :type source: str
        :param output_filename: resulting binary file name
        :type output_filename: str
        :param extraflags: Additional libs to link to
        :type extraflags: list[str]
        :param cross_compile: architecture to compile for or "native"
        :type cross_compiler: str
        :param extraflags: Additional flags to pass to the compiler
        :type extraflags: list[str]
        :return: True if the compilation succeeds, otherwise False
        :rtype: bool
        """
        if self.unsupported:
            raise NotImplementedError(f"{self.executable}: {self.unsupported}")
        cross_flags = []
        if "native" != cross_compile:
            cross_flags = cross_cxx_flags[cross_compile][self.executable] +\
                          cross_lib_flags[cross_compile][self.executable]
        cmd = [self.executable] +\
               self.flags +\
               self.arch_flags +\
               cross_flags +\
               [self.oflag,output_filename] +\
               libs +\
               extraflags +\
               self.stdin_flags

        return self.run_compilation(source=source, output_filename=output_filename, cmd=cmd)
