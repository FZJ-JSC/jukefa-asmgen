# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Calling convention abstraction class
"""

from typing import Callable, Union

from ..asmblocks.noarch import asmgen
from ..asmblocks.op.opmem import opmem_modifier
from ..registers import asm_data_type as adt, greg_base, data_reg

#pylint: disable=too-many-instance-attributes,too-many-positional-arguments


class callconv:
    """
    ISA-independent calling-convention abstraction

    To be used primarily in non-inline ASM, contains methods for
    saving/restoring registers inside a function and when calling a function
    """
    def __init__(self,
                 param_regs : dict[str,list[int]],
                 caller_save_lists : dict[str,list[int]],
                 callee_save_lists : dict[str,list[int]],
                 spreg : int,
                 # Introducing this because x86 params seem
                 # to be 8 bytes off. might turn out to be
                 # an unnecessary hack and the issue is elsewhere
                 spstart : int = 0,
                 spalign : int = 16,
                 ):
        self.param_regs = param_regs
        self.caller_save_lists = caller_save_lists
        self.callee_save_lists = callee_save_lists
        self.spreg = spreg

        self.params : dict[str,tuple[str,int,bool,adt|None]] = {}
        # stack offset for the next parameter
        self.spoff = spstart
        # stack offset for save/restore
        self.spadd = 0
        # stack offset for different register types
        self.sp_rtype_offsets : dict[str,int] = {}

        self.spalign = spalign
        # registers clobbered beyond those the function allocates
        self.clobbers : dict[str,set[int]] = {}

    def add_param(self, type_tag : str, name : str, dt : adt|None = None):
        """
        Add a parameter to the current fuction signature

        :param type_tag: type of register the parameter should be stored in
        :type type_tag: str
        :param name: Name of the parameter
        :type name: str
        :param dt: Data type for freg
        :type dt: class:`asmgen.registers.asm_data_type`|None
        """

        in_stack = False
        if type_tag not in self.param_regs:
            raise ValueError(f"Invalid type tag: {type_tag}")

        if self.param_regs[type_tag]:
            idx = self.param_regs[type_tag].pop(0)
        else:
            idx = self.spoff
            in_stack = True
            self.spoff += 8 # TODO: non-64-bit


        self.params[name] = (type_tag, idx, in_stack, dt)

    def save_regs(self,
                  gen : asmgen,
                  reg_indices : dict[str,list[int]],
                  cfi : bool = False) -> str:
        """
        Returns the string with the necesseray assembly to save the specified
        registers

        :param gen: assembly generator
        :type gen: class:`asmgen.asmblocks.noarch.asmgen`
        :param reg_indices: dict containing lists of register indices to be saved for 
                            each type_tag
        :type reg_indices: dict[str,list[int]]
        :return: string containing the necessary ASM instructions
        :rtype: str
        """

        self.spadd = 8*sum(len(reg_list) for _,reg_list in reg_indices.items())

        asmblock = ""

        spreg = gen.greg(self.spreg)
        if self.spadd > 0:
            aligned_sp = ((self.spadd + self.spalign - 1) // self.spalign) * self.spalign
            asmblock += gen.add(adreg=spreg, bimm=-aligned_sp, cdreg=spreg,
                    a_dt=adt.UINT64, c_dt=adt.UINT64)
            if cfi:
                asmblock += gen.asmwrap(f".cfi_adjust_cfa_offset {aligned_sp}")

        rtype_off = 0
        for type_tag,reg_list in reg_indices.items():
            self.sp_rtype_offsets[type_tag] = rtype_off
            dt = adt.FP64 if 'freg' == type_tag else adt.UINT64

            locations = self.get_locations(
                gen=gen,
                type_tag=type_tag,
                indices=reg_list)
            for i,reg in enumerate(locations):
                offset = 8*i+self.sp_rtype_offsets[type_tag]
                kwargs = {"modifiers": {opmem_modifier.IOFFSET}, "ioffset": offset} if offset else {}
                asmblock += gen.store(dregs=[reg], areg=spreg, dt=dt, **kwargs)
                if cfi:
                    asmblock += gen.asmwrap(f".cfi_rel_offset {reg}, {offset}")
                rtype_off += 8

        return asmblock

    def restore_regs(self,
                     gen : asmgen,
                     reg_indices : dict[str,list[int]],
                  cfi : bool = False) -> str:
        """
        Returns the string with the necessery assembly to restore the specified
        registers (from the stack)

        :param gen: assembly generator
        :type gen: class:`asmgen.asmblocks.noarch.asmgen`
        :param reg_indices: dict containing lists of register indices to be restored
                            for each type_tag
        :type reg_indices: dict[str,list[int]]
        :return: string containing the necessary ASM instructions
        :rtype: str
        """

        spreg = gen.greg(self.spreg)

        asmblock = ""

        for type_tag,reg_list in reg_indices.items():
            dt = adt.FP64 if 'freg' == type_tag else adt.UINT64

            locations = self.get_locations(
                gen=gen,
                type_tag=type_tag,
                indices=reg_list)
            for i,reg in enumerate(locations):
                offset = 8*i+self.sp_rtype_offsets[type_tag]
                kwargs = {"modifiers": {opmem_modifier.IOFFSET}, "ioffset": offset} if offset else {}
                asmblock += gen.load(dregs=[reg], areg=spreg, dt=dt, **kwargs)
                if cfi:
                    asmblock += gen.asmwrap(f".cfi_restore {reg}")

        if self.spadd > 0:
            aligned_sp = ((self.spadd + self.spalign - 1) // self.spalign) * self.spalign
            asmblock += gen.add(adreg=spreg, bimm=aligned_sp, cdreg=spreg,
                    a_dt=adt.UINT64, c_dt=adt.UINT64)
            if cfi:
                asmblock += gen.asmwrap(f".cfi_adjust_cfa_offset -{aligned_sp}")

        self.spadd = 0

        return asmblock


    def get_locations(self,
                      gen : asmgen,
                      type_tag : str,
                      indices : list[int]) -> list[Union[greg_base,data_reg]]:
        """
        Returns the registers corresponding to the specified register indices
        """

        kwargs = {
        }
        if 'freg' == type_tag:
            kwargs['dt'] = adt.FP64

        return [getattr(gen,type_tag)(**(kwargs|{'reg_idx':idx}))\
                for idx in indices]

    def save_or_restore(self,
             gen : asmgen,
             regs : dict[str,list[int]],
             save_lists : dict[str,list[int]],
             action : Callable[..., str],
             cfi : bool = False) -> str:
        """
        Abstraction over generating the ASM for saving and restoring registers

        :param gen: ASM generator
        :type gen: class:`asmgen.asmblocks.noarch.asmgen`
        :param regs: lists of reg indices to selectively save/restore for each type_tag
        :type regs: dict[str,list[int]]
        :param save_lists: lists of reg indices that require saving for each type tag
        :type save_lists: dict[str,list[int]]
        :param action: the function (save/restore) to apply to the selected registers
        :type action: Callable[[class:`asmgen.asmblocks.noarch.asmgen`,dict[str,list[int]]],str]
        :return: string containing the necessary ASM instructions
        :rtype: str
        """

        reg_indices : dict[str,list[int]] = {
                k : [] for k in save_lists.keys()
                }

        for tag in save_lists.keys():
            if tag not in regs:
                continue
            reg_indices[tag] = set(regs[tag]).intersection(
                    set(save_lists[tag]))

        return action(gen=gen, reg_indices=reg_indices, cfi=cfi)

    def save_before_call(self,
                      gen : asmgen,
                      regs : dict[str,list[int]],
            cfi : bool = False) -> str:
        """
        Returns a string containing instructions to save registers to stack that are
        the resposibility of the caller

        :param gen: ASM generator
        :type gen: class:`asmgen.asmblocks.noarch.asmgen`
        :param regs: lists of reg indices to selectively save for each type_tag
        :type regs: dict[str,list[int]]

        :return: string containing the necessary ASM instructions
        :rtype: str
        """

        return self.save_or_restore(
                gen=gen,
                regs=regs,
                save_lists=self.caller_save_lists,
                action=self.save_regs,
                cfi=cfi)

    def restore_after_call(self,
                           gen : asmgen,
                           regs : dict[str,list[int]],
            cfi : bool = False) -> str:
        """
        Returns a string containing instructions to restore registers from stack that
        are the resposibility of the caller

        :param gen: ASM generator
        :type gen: class:`asmgen.asmblocks.noarch.asmgen`
        :param regs: lists of reg indices to selectively restore for each type_tag
        :type regs: dict[str,list[int]]

        :return: string containing the necessary ASM instructions
        :rtype: str
        """
        return self.save_or_restore(
                gen=gen,
                regs=regs,
                save_lists=self.caller_save_lists,
                action=self.restore_regs,
                cfi=cfi)

    def add_clobber(self, type_tag : str, indices) -> None:
        """
        Registers the function clobbers beyond those it allocates - an
        instruction's side effect (SME's smstart and smstop zero the vector
        registers). Saved and restored in the call where the convention makes
        them the callee's: checked against its callee-saved lists here
        :param type_tag: register type, as in the save lists ('greg', 'freg')
        :param indices: the registers' numbers
        """
        self.clobbers.setdefault(type_tag, set()).update(indices)

    def deprioritize_callee_saved(self, rt, priority : int = -1):
        """
        Lowers the allocation priority of the registers the convention makes
        the callee's: general allocation takes the caller's first, sparing a
        save and restore. The tracker gives a register's priority to its
        counterparts in a shared pool - AArch64's v8-v15 where d8-d15 are
        the callee's
        :param rt: the register tracker
        :param priority: the priority given, where not already lower
        """
        for tag, regs in self.callee_save_lists.items():
            if tag not in getattr(rt, "registered_types", ()):
                continue
            for idx in regs:
                if idx < rt.max_regs[tag]:
                    rt.set_priority(tag, idx, min(priority, rt.priority(tag, idx)))

    def _with_clobbers(self, regs : dict[str,list[int]]) -> dict[str,list[int]]:
        tags = set(regs) | set(self.clobbers)
        return {t: sorted(set(regs.get(t, [])) | self.clobbers.get(t, set())) for t in tags}

    def save_in_call(self,
                     gen : asmgen,
                     regs : dict[str,list[int]],
            cfi : bool = False) -> str:
        """
        Returns a string containing instructions to save registers to stack that are
        the resposibility of the callee

        :param gen: ASM generator
        :type gen: class:`asmgen.asmblocks.noarch.asmgen`
        :param regs: lists of reg indices to selectively save for each type_tag
        :type regs: dict[str,list[int]]

        :return: string containing the necessary ASM instructions
        :rtype: str
        """
        return self.save_or_restore(
                gen=gen,
                regs=self._with_clobbers(regs),
                save_lists=self.callee_save_lists,
                action=self.save_regs,
                cfi=cfi)

    def restore_before_ret(self,
                     gen : asmgen,
                     regs : dict[str,list[int]],
            cfi : bool = False) -> str:
        """
        Returns a string containing instructions to restore registers from stack that
        are the resposibility of the callee

        :param gen: ASM generator
        :type gen: class:`asmgen.asmblocks.noarch.asmgen`
        :param regs: lists of reg indices to selectively restore for each type_tag
        :type regs: dict[str,list[int]]

        :return: string containing the necessary ASM instructions
        :rtype: str
        """
        return self.save_or_restore(
                gen=gen,
                regs=self._with_clobbers(regs),
                save_lists=self.callee_save_lists,
                action=self.restore_regs,
                cfi=cfi)

    def get_params(self):
        """
        return the internal structure containing the current function parameters
        """
        return self.params
