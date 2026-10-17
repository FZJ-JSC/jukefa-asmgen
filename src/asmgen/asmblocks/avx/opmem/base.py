# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
AVX operations with n data operand and 1 address operand (load/store)
"""
from typing import Callable, Any

from ...op import (
    opmem,
    opmem_modifier as mod,
    opmem_action,
    operation_signature,
    operand_modifier as opd_mod
)
from ....registers import (
    asm_data_type as adt,
    adt_size,
    ait_size,
    data_reg
)
from ..types import (
    x86_greg,
    avx_vreg, xmm_vreg, ymm_vreg, zmm_vreg,
    avx_freg,
    avx512_mreg
)
from ...x86.opmem.base import x86_opmem

from .signatures import make_avx_opmem_signatures

class avx_opmem(opmem):
    """
    AVX instruction with n data operands and 1 address operand.
    Uses x86_opmem to automatically handle scalar routing.
    """
    def __init__(self, action: opmem_action, simd_bytes: int,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        self.action = action
        self.simd_bytes = simd_bytes
        self.asmwrap = asmwrap
        self.rpref = rpref
        self.scalar_opmem = x86_opmem(action=action, asmwrap=asmwrap, rpref=rpref)

        self.signatures = []

    def get_signatures(self) -> list[operation_signature]:
        return self.signatures

    def diagnose_unsupported_mods(self, modifiers: set[mod]):
        """
        Check if any modifier is unsupported at all
        """

        unsupported_mods = {
            mod.TINDEX:  (ValueError, "Base X86_64 has no ld/st with 2D tile offset indices"),
            mod.VOFFSET: (ValueError, "Base X86_64 has no ld/st with 2D tile offsets"),
            mod.TOFFSET: (ValueError, "Base X86_64 has no ld/st with 2D tile offsets"),
            mod.ISTRIDE: (ValueError, "Base X86_64 has no ld/st with immediate strides"),
            mod.GSTRIDE: (ValueError, "Base X86_64 has no ld/st with GP-reg strides"),
            mod.STRUCT:  (ValueError, "Base X86_64 has no structured ld/st"),
            mod.POSTINC: (ValueError, "Base X86_64 has no postinc ld/st"),
            mod.NT:      (ValueError, "Base X86_64 has no non-temporals ld/st"),
        }

        for m, (exc_type, msg) in unsupported_mods.items():
            if m in modifiers:
                raise exc_type(msg)

    def diagnose_unsupported_opd_mods(self, opd_mods : dict[str,set[opd_mod]]):
        """
        Check if any operand modifier is unsupported at all
        """

        unsupported_opd_mods = {
            opd_mod.VF:        (ValueError, "VF mod makes no sense for ld/st"),
            opd_mod.BLOCKLANE: (ValueError, "BLOCKLANE makes no sense for ld/st"),
            opd_mod.ROW:       (ValueError, "Base X86_64 has no row selection ld/st"),
            opd_mod.COL:       (ValueError, "Base X86_64 has no column selection ld/st"),
            opd_mod.GLANE:     (ValueError, "Base X86_64 has no GP-reg lane selection ld/st"),
        }
        for umod, (exc_type, msg) in unsupported_opd_mods.items():
            for _, mods in opd_mods.items():
                if umod in mods:
                    raise exc_type(msg)

    def diagnose_missing_params(self, modifiers : set[mod],
                                operand_modifiers: dict[str,set[opd_mod]],
                                kwargs : dict[str,Any]):
        """
        Check if mandatory additional parameters for specific modifiers are missing
        """

        required_params = {
            mod.IOFFSET : ['ioffset'],
            mod.VOFFSET : ['voffset'],
            mod.VINDEX  : ['vidxreg','it'],
            mod.GOFFSET : ['offreg'],
        }
        for m, plist in required_params.items():
            for p in plist:
                if m in modifiers and p not in kwargs:
                    raise ValueError(f"{m.name} modifier requires '{p}' parameter")

        required_opd_params = {
            opd_mod.ILANE   : ['lane'],
        }

        for m, plist in required_opd_params.items():
            for p in plist:
                for opd, mods in operand_modifiers.items():
                    p_opd = f"{opd}_{p}"
                    if m in mods and  p_opd not in kwargs:
                        raise ValueError(
                                f"{m.name} modifier for {opd} requires '{p_opd}' parameter")

    def diagnose_failure(self, modifiers : set[mod],
                         operand_modifiers : dict[str,set[opd_mod]],
                         kwargs : dict[str,Any],
                         dts : dict[str, adt]) -> list[operation_signature]:

        self.diagnose_unsupported_mods(modifiers)
        self.diagnose_unsupported_opd_mods(operand_modifiers)
        self.diagnose_missing_params(modifiers, operand_modifiers, kwargs)


    def get_addressing(self, areg: x86_greg, modifiers: set[mod], **kwargs) -> str:
        """
        Constructs the addressing string

        :param areg: address register
        :param modifiers: operation modifiers

        :return: string containing the addressing
        """
        offset = 0
        if mod.IOFFSET in modifiers:
            offset = kwargs["ioffset"]
        elif mod.VOFFSET in modifiers:
            offset = kwargs["voffset"] * self.simd_bytes

        pareg = self.rpref(areg)

        return f"{offset}({pareg})" if offset != 0 else f"({pareg})"

    def get_vector_mnemonic(self, dt: adt) -> str:
        """
        Get type dependent mnemonic

        :param dt: data type
        """
        if dt == adt.FP32:
            return "vmovups"
        if dt == adt.FP64:
            return "vmovupd"
        return "vmovdqu"


    def build_bcast(self, dreg: avx_vreg, areg: x86_greg, dt: adt,
                    addressing: str) -> str:
        """
        Generate broadcast instruction

        :param dreg: vector register to broadcast into
        :param areg: gp register with the base address
        :param dt: element data type
        :param addressing: string containing the addressing
        """
        raise NotImplementedError("Missing broadcast implementation")

    def build_gather(self, dreg: avx_vreg, mreg: avx_vreg|avx512_mreg,
                     areg: x86_greg, dt:adt, **kwargs) -> str:
        """
        Generate gather instruction

        :param dreg: vector register to gather into
        :param areg: gp register with the base address
        :param mreg: mask register (xmmN/ymmN for AVX2, kN for AVX512)
        :param dt: element data type
        """
        raise NotImplementedError("Missing gather implementation")

    def build_scatter(self, dreg: avx_vreg, mreg: avx_vreg|avx512_mreg,
                      areg: x86_greg, dt:adt, **kwargs) -> str:
        """
        Generate scatter instruction

        :param dreg: vector register to scatter from
        :param areg: gp register with the base address
        :param mreg: mask register (xmmN/ymmN for AVX2, kN for AVX512)
        :param dt: element data type
        """
        raise NotImplementedError("Missing scatter implementation")

    def build_lane_load(self, dreg: avx_vreg, areg: x86_greg, dt: adt,
                        lane: int, addressing: str) -> str:
        """
        Generate scalar load into a vector lane

        :param dreg: vector register to load into
        :param areg: gp register with the base address
        :param dt: element data type
        :param lane: vector lane/element index
        """
        raise NotImplementedError("Missing lane load implementation")

    def build_lane_store(self, dreg: avx_vreg, areg: x86_greg, dt: adt,
                        lane: int, addressing: str) -> str:
        """
        Generate scalar store from a vector lane

        :param dreg: vector register to store from
        :param areg: gp register with the base address
        :param dt: element data type
        :param lane: vector lane/element index
        """
        raise NotImplementedError("Missing lane store implementation")

    def build_vindex(self, dreg : avx_vreg, mreg : avx_vreg|avx512_mreg,
                     areg : x86_greg, dt : adt, **kwargs) -> str:
        """
        Build gather/scatter

        :param dreg: vector register to use
        :param mreg: mask register to use
        :param areg: gp register with the base address
        :param dt: element data type
        """
        if self.action == opmem_action.LOAD:
            return self.asmwrap(
                    self.build_gather(dreg, mreg, areg, dt, **kwargs))
        if self.action == opmem_action.STORE:
            return self.asmwrap(
                    self.build_scatter(dreg, mreg, areg, dt, **kwargs))

        # Potentially adding prefetches or something else in the future
        raise ValueError(f"Action {self.action} with VINDEX not valid")


    # It's fine
    # pylint: disable-next=too-many-return-statements
    def implementation(self, *, dregs: list[data_reg], agreg: x86_greg,
                       modifiers: set[mod],
                       operand_modifiers : dict[str,set[opd_mod]],
                       **kwargs) -> str:
        if not dregs:
            raise ValueError("No dregs provided")

        a_dt = kwargs.pop('a_dt')

        if isinstance(dregs[0], (x86_greg, avx_freg)):
            return self.scalar_opmem(dregs=dregs, areg=agreg, dt=a_dt,
                                      modifiers=modifiers,
                                      operand_modifiers=operand_modifiers,
                                      **kwargs)

        dreg = dregs[0]

        addressing = self.get_addressing(agreg, modifiers, **kwargs)


        has_bcast = any(opd_mod.BCAST in mods for mods in operand_modifiers.values())
        if has_bcast:
            return self.asmwrap(self.build_bcast(dreg, agreg, a_dt, addressing))

        if mod.VINDEX in modifiers:
            return self.build_vindex(dreg, kwargs['amreg'], agreg, a_dt, **kwargs)

        has_ilane = any(opd_mod.ILANE in mods for mods in operand_modifiers.values())
        if has_ilane:
            lane = kwargs["adreg_lane"]
            if not isinstance(dreg, xmm_vreg):
                xmm_alias = xmm_vreg(dreg.idx)
            else:
                xmm_alias = dreg

            if self.action == opmem_action.LOAD:
                return self.asmwrap(
                        self.build_lane_load(xmm_alias, agreg, a_dt, lane, addressing))
            if self.action == opmem_action.STORE:
                return self.asmwrap(
                        self.build_lane_store(xmm_alias, agreg, a_dt, lane, addressing))

            raise ValueError(f"Action {self.action} with ILANE not valid")

        pdreg = self.rpref(dreg)
        inst = self.get_vector_mnemonic(a_dt)
        if self.action == opmem_action.LOAD:
            return self.asmwrap(f"{inst} {addressing}, {pdreg}")
        if self.action == opmem_action.STORE:
            return self.asmwrap(f"{inst} {pdreg}, {addressing}")

        raise ValueError(f"Action {self.action} not valid")


class avx128_opmem(avx_opmem):
    """
    AVX2 128bit opmem implementation
    """
    ie_suffixes = {1:'b',2:'w',4:'d',8:'q'}
    def __init__(self, action: opmem_action,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action, simd_bytes=16, asmwrap=asmwrap, rpref=rpref)

        self.signatures = make_avx_opmem_signatures(action=self.action, is_avx512=False)
        self.signatures.extend(self.scalar_opmem.get_signatures())

    def build_lane_load(self, dreg: xmm_vreg, areg: x86_greg, dt: adt,
                        lane: int, addressing: str) -> str:
        size = adt_size(dt)
        pdreg = self.rpref(dreg)

        if dt == adt.FP64:
            if lane == 0:
                return f"vmovsd {addressing}, {pdreg}"
            if lane == 1:
                return f"vmovhpd {addressing}, {pdreg}, {pdreg}"
            raise ValueError(f"FP64 lane {lane} out of bounds")
        # Float 32 uses vinsertps (syntax is slightly quirky:
        # imm8 dictates dst/src/zero mask)
        # Imm8 format for memory load: (lane << 4)
        if dt == adt.FP32:
            if lane == 0:
                return f"vmovss {addressing}, {pdreg}"
            imm8 = lane << 4
            return f"vinsertps ${imm8}, {addressing}, {pdreg}, {pdreg}"

        # Everything else uses integer domain inserts (vpinsrb/w/d/q)

        return f"vpinsr{self.ie_suffixes[size]} ${lane}, {addressing}, {pdreg}, {pdreg}"

    def build_lane_store(self, dreg: xmm_vreg, areg: x86_greg, dt: adt,
                         lane: int, addressing: str) -> str:
        size = adt_size(dt)
        pdreg = self.rpref(dreg)

        if dt == adt.FP64:
            if lane == 0:
                return f"vmovsd {pdreg},{addressing}"
            if lane == 1:
                return f"vmovhpd {pdreg},{addressing}"
            raise ValueError(f"FP64 lane {lane} out of bounds")

        # Float 32 uses extractps
        if dt == adt.FP32:
            imm8 = lane << 4
            return f"vextractps ${imm8}, {pdreg}, {addressing}"

        # Integer domain extracts (vpextrb/w/d/q)
        return f"vpextr{self.ie_suffixes[size]} ${lane}, {pdreg}, {addressing}"

    def build_bcast(self, dreg: xmm_vreg, areg: x86_greg, dt: adt, addressing: str) -> str:
        suf = "ss" if adt_size(dt) == 4 else "sd"
        pdreg = self.rpref(dreg)
        return f"vbroadcast{suf} {addressing}, {pdreg}"

    def build_gather(self, dreg: xmm_vreg|ymm_vreg, mreg : xmm_vreg|ymm_vreg,
                     areg: x86_greg, dt: adt, **kwargs) -> str:
        suf = "ps" if adt_size(dt) == 4 else "pd"
        isuf = "d" if ait_size(kwargs["it"]) == 4 else "q"
        vidxreg = kwargs["vidxreg"]

        pareg = self.rpref(areg)
        pvidxreg = self.rpref(vidxreg)
        pdreg = self.rpref(dreg)
        pmreg = self.rpref(mreg)

        return f"vgather{isuf}{suf} {pmreg}, ({pareg},{pvidxreg},1), {pdreg}"

    def build_scatter(self, dreg: xmm_vreg, mreg : xmm_vreg,
                      areg: x86_greg, dt: adt, **kwargs) -> str:
        raise NotImplementedError("AVX/AVX2 has no store with vector register stride (Scatter)")

class avx256_opmem(avx128_opmem):
    """
    AVX2 256bit opmem implementation
    """
    def __init__(self, action: opmem_action,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        # AVX256 is functionally identical to 128 here, just with 32-byte offsets
        super().__init__(action=action, asmwrap=asmwrap, rpref=rpref)
        self.simd_bytes = 32

        self.signatures = make_avx_opmem_signatures(action=self.action, is_avx512=False)
        self.signatures.extend(self.scalar_opmem.get_signatures())

class avx512_opmem(avx_opmem):
    """
    AVX512 opmem implementation
    """
    def __init__(self, action: opmem_action,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action, simd_bytes=64, asmwrap=asmwrap, rpref=rpref)

        self.signatures = make_avx_opmem_signatures(action=self.action, is_avx512=True)
        self.signatures.extend(self.scalar_opmem.get_signatures())

    def build_bcast(self, dreg: zmm_vreg, areg: x86_greg, dt: adt, addressing: str) -> str:
        suf = "ss" if adt_size(dt) == 4 else "sd"

        pdreg = self.rpref(dreg)
        return f"vbroadcast{suf} {addressing}, {pdreg}"

    def build_gather(self, dreg: zmm_vreg, mreg : avx512_mreg,
                     areg: x86_greg, dt: adt, **kwargs) -> str:
        suf = "ps" if adt_size(dt) == 4 else "pd"
        isuf = "d" if ait_size(kwargs["it"]) == 4 else "q"
        vidxreg = kwargs["vidxreg"]

        pareg = self.rpref(areg)
        pvidxreg = self.rpref(vidxreg)

        addressing = f"({pareg},{pvidxreg},1)"
        maskreg = self.rpref(mreg)
        #masksuf = "w" if adt_size(dt) == 4 else "q"

        pdreg = self.rpref(dreg)

        #return (f"kxnor{masksuf} {maskreg}, {maskreg}, {maskreg}\n"
        #        f"vgather{isuf}{suf} {addressing}, {pdreg}{{{maskreg}}}")
        return f"vgather{isuf}{suf} {addressing}, {pdreg}{{{maskreg}}}"

    def build_scatter(self, dreg: zmm_vreg, mreg: avx512_mreg,
                      areg: x86_greg, dt: adt, **kwargs) -> str:
        suf = "ps" if adt_size(dt) == 4 else "pd"
        isuf = "d" if ait_size(kwargs["it"]) == 4 else "q"
        vidxreg = kwargs["vidxreg"]

        pareg = self.rpref(areg)
        pvidxreg = self.rpref(vidxreg)

        addressing = f"({pareg},{pvidxreg},1)"
        maskreg = self.rpref(mreg)
        #masksuf = "w" if adt_size(dt) == 4 else "q"

        pdreg = self.rpref(dreg)

        #return (f"kxnor{masksuf} {maskreg}, {maskreg}, {maskreg}\n"
        #        f"vscatter{isuf}{suf} {dreg}, {addressing}{{{maskreg}}}")
        return f"vscatter{isuf}{suf} {pdreg}, {addressing}{{{maskreg}}}"
