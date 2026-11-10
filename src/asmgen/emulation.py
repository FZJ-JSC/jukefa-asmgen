# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Utilities for running emulations of different ISAs on other ISAs
"""
import logging
import os
import platform
import re
import shutil
from itertools import permutations
from subprocess import PIPE, Popen

from .compilation.compiler_presets import cross_archs


class emulator:
    """
    Base emulator class
    """
    def __init__(self):
        self.exe = "invalid"
        self.flags = []

    def run(self, executable : str):
        """
        Runs an executable through this emulator

        :param executable: binary to execute
        :type executable: str
        :raises RuntimeError: If the execution fails or doesn't return 0
        """
        log = logging.getLogger("EMULATOR")
        cmd = [self.exe]+self.flags+[executable]
        output = ""
        errout = ""
        rcode = 0
        with Popen(cmd,stdin=PIPE,stdout=PIPE,stderr=PIPE) as process:
            process_out = process.communicate()
            output = process_out[0].decode()
            errout = process_out[1].decode()
            rcode = process.returncode
        log.debug("Execution stdout: %s", output)
        if 0 != rcode:
            log.error("Execution stderr: %s", errout)
            raise RuntimeError(f"Error occurred running {executable}")
        log.debug("Execution stderr: %s", errout)

class sde_static_emulator(emulator):
    """
    Emulates with Intel SDE
    """
    # explicitly not calling the super init
    # pylint: disable=super-init-not-called
    def __init__(self):
        self.exe = "sde"
        self.flags = ["-future","--"]


class qemu_static_emulator(emulator):
    """
    Emulates with QEMU
    """
    # explicitly not calling the super init
    # pylint: disable=super-init-not-called
    def __init__(self, arch : str, cpuargs : str):
        """
        Constructor method

        :param arch: target architecture (x86_64,aarch64...)
        :type arch: str
        :param cpuargs: string to pass to qemu's '-cpu' argument
        :type cpuargs: str
        """
        suffix_permutations = set(permutations(["static",arch])) |\
                              set(permutations(["static","user",arch]))

        possible_exe_names = ["qemu-" + "-".join(p) for p in suffix_permutations]

        self.exe = ""
        for name in possible_exe_names:
            executable = shutil.which(name)
            if None is not executable:
                self.exe = executable
                break
        if not self.exe:
            raise RuntimeError(f"Could not find a static qemu that supports {arch} architecture")

        # Find library dir
        possible_roots = [
                f"/usr/{arch}-pc-linux-gnu",
                f"/usr/local/{arch}-pc-linux-gnu",
                f"/usr/{arch}-linux-gnu",
                f"/usr/local/{arch}-linux-gnu",
                f"/usr/{arch}-unknown-gnu",
                f"/usr/local/{arch}-unknown-gnu",
                ]


        arch_upper = arch.upper()
        more_roots_str = os.getenv(f"{arch_upper}_CROSS_ROOT_PATH",default=None)
        if None is not more_roots_str:
            more_roots = more_roots_str.split(":")
            possible_roots = [os.path.join(r,f"{arch}-linux-gnu") for r in more_roots]+\
                             [os.path.join(r,f"{arch}-pc-linux-gnu") for r in more_roots]+\
                             [os.path.join(r,f"{arch}-unknown-gnu") for r in more_roots]+\
                             possible_roots

        if platform.machine() == arch:
            self.root = "/usr"
        else:
            for r in possible_roots:
                if os.path.exists(r) and os.path.isdir(r):
                    self.root = r
                    break

        self.cpuargs = cpuargs

        self.flags = [
                "-L",self.root,
                "-E",("LD_LIBRARY_PATH=" 
                      f"{os.path.join(self.root,'lib')}:"
                      f"{os.path.join(self.root,'lib64')}"),
                "-cpu",self.cpuargs
                ]


emulators = {
        ("x86_64", "fma128")  : "qemu-x86_64-static>7.2",
        ("x86_64", "fma256")  : "qemu-x86_64-static>7.2",
        ("x86_64", "avx512")  : "sde",
        ("x86_64", "neon")    : "qemu-aarch64-static",
        ("x86_64", "sve")     : "qemu-aarch64-static",
        ("x86_64", "sme")     : "qemu-aarch64-static",
        ("x86_64", "rvv")     : "qemu-riscv64-static",
        ("x86_64", "rvv071")  : "rave",
        ("aarch64", "fma128") : "qemu-x86_64-static>7.2",
        ("aarch64", "fma256") : "qemu-x86_64-static>7.2",
        ("aarch64", "neon")   : "qemu-aarch64-static",
        ("aarch64", "sve")    : "qemu-aarch64-static",
        ("aarch64", "sme")    : "qemu-aarch64-static",
        ("aarch64", "rvv")    : "qemu-riscv64-static",
        ("riscv64", "fma128") : "qemu-x86_64-static>7.2",
        ("riscv64", "fma256") : "qemu-x86_64-static>7.2",
        ("riscv64", "neon")   : "qemu-aarch64-static",
        ("riscv64", "sve")    : "qemu-aarch64-static",
        ("riscv64", "sme")    : "qemu-aarch64-static",
        ("riscv64", "rvv")    : "rave",
        ("riscv64", "rvv071") : "rave",
        }

qemu_cpu_flags = {
        "rvv071" : "max,vlen=512,vext_spec=v0.7.1",
        "rvv" : "max,vlen=512,vext_spec=v1.0",
        "neon" : "max",
        "sve" : "max,sve=on,sve512=on",
        "sme" : "max,sve=on,sve512=on,sme512=on",
        "fma128" : "max",
        "fma256" : "max",
        "avx512" : "max",
}

def get_emulator(host_arch : str, target_isa : str) -> emulator:
    """
    Returns an emulator that can run target ISA executables on the host arch

    :param host_arch: Host architecture
    :type host_arch: str
    :param target_isa: Target asmgen/SIMD ISA
    :type target_isa: str
    :return: emulator that will work for requested ISA on the current host
    :rtype: class:`asmgen.emulation.emulator`
    """
    emulator_spec = emulators[(host_arch, target_isa)]
    # This looks like an anime face lol
    executable = re.split(">=<", emulator_spec)[0]

    if "qemu" in executable:
        return qemu_static_emulator(arch=cross_archs[target_isa],
                                    cpuargs=qemu_cpu_flags[target_isa])
    if "sde" in executable:
        return sde_static_emulator()

    raise NotImplementedError("No Emulator known for the specified combination")
