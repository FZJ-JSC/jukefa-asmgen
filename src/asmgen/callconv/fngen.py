# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Utilities for generating C-callable functions in ASM
"""

from .callconv import callconv
from ..asmblocks.op.opmem import opmem_modifier
from ..registers import asm_data_type as adt
from ..asmblocks.noarch import asmgen
from ..registers import reg_tracker

class fngen:
    """
    Class for generating calling-convention-compliant functions in ASM
    """

    def __init__(self, gen : asmgen, rt : reg_tracker | None = None, *,
                 cfi : bool | None = None):
        self.gen = gen
        self.rt = rt
        self._cfi = cfi

        self.required_loads : dict[str,tuple[str,int]] = dict()

        self.debug_block : str = ""

    def function(self, name : str, body : str, *, align : int | None = None,
                 offset : int = 0, data : str = "") -> str:
        """
        The function in assembly: its section and alignment, .global and
        .type, the label, the body - its instructions through the return -
        .size, then the data it keeps (constants in their own section). What
        the file says once, gen.asmfile_annotations, is apart from it.

        :param name: the function's symbol
        :param body: its instructions, saves to return
        :param align: the function's alignment in bytes, if any
        :param offset: where it starts past that boundary, in bytes (nops)
        :param data: its data, after it
        """
        gen = self.gen
        text = gen.asmwrap(".text")
        if align:
            text += gen.align(align, offset)
        text += gen.asmwrap(f".global {name}")
        text += gen.asmwrap(f".type {name}, %function")
        text += gen.asmwrap(f"{name}:")
        if self.cfi:
            body = gen.asmwrap(".cfi_startproc") + body + gen.asmwrap(".cfi_endproc")
        return text + body + gen.asmwrap(f".size {name}, .-{name}") + data

    @property
    def cfi(self) -> bool:
        """
        Whether functions carry unwind information (CFI): .cfi_startproc and
        .cfi_endproc around them (function), the stack adjustment and where
        each register is saved after the saves (get_boilerplate). By default
        where the output is a file of its own - inline assembly's CFI is the
        compiler's. fngen(..., cfi=) says otherwise.
        """
        return (not self.gen.output_inline) if self._cfi is None else self._cfi

    def init_cc(self, cc : callconv,
                reverse_alias_map : dict[str,str] = None,
                unused_parameters : set[str] = None):
        """
        Initialize the function wrapper according to calling convention

        :param cc: calling convention
        :param reverse_alias_map: maps aliases onto the actual parameters
        :param unused_parameters: parameters that aren't used in the function
                                  therefore don't need register allocation
        """

        if reverse_alias_map is None:
            reverse_alias_map = dict()

        if unused_parameters is None:
            unused_parameters = set()

        params = cc.get_params()
        for name,(tag,idx,in_stack,dt) in params.items():
            if name in reverse_alias_map:
                name = reverse_alias_map[name]
            if name in unused_parameters:
                continue
            if not in_stack:
                self.rt.reserve_specific_reg(tag, idx)
                self.rt.alias_reg(tag, name, idx)
            else:
                regidx = self.rt.reserve_any_reg(tag)
                self.rt.alias_reg(tag, name, regidx)
                self.required_loads[name] = (tag, idx)

        self.debug_block = ""
        for name,(tag,_,_,dt) in params.items():
            if name in reverse_alias_map:
                name = reverse_alias_map[name]
            if name in unused_parameters:
                continue

            kwargs = {
                'reg_idx' : self.rt.aliased_regs[tag][name]
            }
            if dt is not None:
                kwargs['dt'] = dt
            reg = getattr(self.gen, tag)(**kwargs)
            self.debug_block += self.gen.asmwrap(f"# {reg} = {name}")

    def get_boilerplate(self,
                        cc : callconv,
                        unused_parameters : set[str] = None) -> tuple[str,str,str]:
        """
        Get function boilerplate (instruction block for saving registers, instruction block for
        loading parameters and an instruction block for restoring previously saved registers)

        :param cc: calling convention
        :param unused_parameters: parameters that aren't used in the function
                                  therefore don't need register allocation
        :return: Tuple consisting of a saveblock,loadblock and restoreblock
        """
        if unused_parameters is None:
            unused_parameters = set()

        used_gregs = self.rt.get_clobbered_regs('greg')
        # an FP register is clobbered where the vector register it's part of
        # is (AArch64's d8 in v8): the tracker knows the files it shares
        used_fregs = list(self.rt.get_clobbered_regs('freg', through_shared=True))

        sr_count = len(set(used_gregs).intersection(cc.callee_save_lists['greg']))
        sr_count += len(set(used_fregs).intersection(cc.callee_save_lists['freg']))
        saveblock = cc.save_in_call(self.gen,
                                    regs={'greg':used_gregs,
                                          'freg':used_fregs},
                                    cfi=self.cfi)

        loadblock = ''

        # the save area as the convention laid it out - with the registers
        # added as clobbers (add_clobber), not only those allocated
        raw_sp_shift = cc.spadd
        aligned_sp_shift = ((raw_sp_shift + cc.spalign - 1) // cc.spalign) * cc.spalign
        # where stack parameters are, from the stack pointer after the save:
        # for loads of them emitted elsewhere than at entry
        self.sp_shift = aligned_sp_shift

        for name,(tag,off) in self.required_loads.items():
            loadblock += self.gen.asmwrap(f"# loading {name}")
            offset = off+aligned_sp_shift # Assuming 64 bit/ 8 byte pointers
            kwargs = {"modifiers": {opmem_modifier.IOFFSET}, "ioffset": offset} if offset else {}
            loadblock += self.gen.load(dregs=[self.gen.greg(self.rt.aliased_regs[tag][name])],
                                       areg=self.gen.greg(cc.spreg), dt=adt.UINT64, **kwargs)


        restoreblock = cc.restore_before_ret(self.gen,
                                             regs={'greg':used_gregs,
                                                   'freg':used_fregs},
                                             cfi=self.cfi)

        return saveblock,self.debug_block+loadblock,restoreblock
