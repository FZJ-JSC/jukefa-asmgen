# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
AVX register types (vector, masking registers and AT&T prefixing)
"""
from typing import Callable

from ...registers import vreg_base, greg_base, data_reg, mreg_base

from ..x86.types import x86_greg, avx_freg

class avx_vreg(vreg_base):
    """
    AVX base vector register
    """
    def __init__(self, reg_idx : int):
        self.reg_idx = reg_idx

    @property
    def idx(self) -> int:
        return self.reg_idx

    def __str__(self) -> str:
        raise NotImplementedError("Base avx_vreg used (use {x,y,z}mm_vreg instead)")

class xmm_vreg(avx_vreg):
    """
    AVX 128 bit vector register (xmm)
    """
    def __init__(self, reg_idx : int):
        super().__init__(reg_idx=reg_idx)

    def __str__(self) -> str:
        return f"xmm{self.idx}"

class ymm_vreg(avx_vreg):
    """
    AVX 256 bit vector register (ymm)
    """
    def __init__(self, reg_idx : int):
        super().__init__(reg_idx=reg_idx)

    def __str__(self) -> str:
        return f"ymm{self.idx}"

class zmm_vreg(avx_vreg):
    """
    AVX512 vector register (zmm)
    """
    def __init__(self, reg_idx : int):
        super().__init__(reg_idx=reg_idx)

    def __str__(self) -> str:
        return f"zmm{self.idx}"

class avx512_mreg(mreg_base):
    """
    AVX512 mask register (k0-k7)
    """
    def __init__(self, reg_idx : int):
        if 0 > reg_idx or 7 < reg_idx:
            raise ValueError(f"mask reg idx must be 0 <= idx <= 7 (is: {reg_idx})")
        self.reg_idx = reg_idx

    @property
    def idx(self) -> int:
        return self.reg_idx

    def __str__(self) -> str:
        return f"k{self.idx}"


class reg_prefixer:
    """
    Helper class for prefixing a register with '%%' in inline ASM and '%' in normal ASM

    This ensures the registers are refered to correctly in AT&T/GAS style syntax
    """

    def __init__(self, output_inline_getter : Callable[[],bool]):
        self.output_inline_getter = output_inline_getter

    @property
    def output_inline(self) -> bool:
        """
        Returns True if inline ASM output is active        
        """
        return self.output_inline_getter()

    def __call__(self, reg : data_reg|greg_base|avx512_mreg,
                 size : int = 8) -> str:
        """
        Depending on inline output state, prepends the string representation of the
        register with a '%%' for inline ASM and '%' for normal ASM
        """

        if "rip" == reg:
            # the instruction pointer: RIP-relative addresses
            regstr = "rip"
        elif not isinstance(reg, (x86_greg, avx_freg,
                                  xmm_vreg, ymm_vreg, zmm_vreg,
                                  avx512_mreg)):
            raise ValueError(f"{reg} is not a x86 or AVX register")
        else:
            regstr = str(reg)
        if isinstance(reg, x86_greg):
            regstr = reg.name(size)
        # If there is a [ it's probably a parameter
        if '[' in regstr:
            return regstr
        pref = '%%' if self.output_inline else '%'
        return f"{pref}{regstr}"
