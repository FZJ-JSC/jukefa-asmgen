# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Valid signatures for AVX opmem operations
"""

from ...x86.opmem.displacement import displacement_forms
from ...op.constraint import multiple_constraint
from ...op import (
    operation_signature as sig,
    operand_shape as osh,
    operand_type as ot,
    operand_role as orl,
    operand_modifier as opd_mod,
    register_type as rt,
    opmem_modifier as mod,
    opmem_action
)

from ...op.constraint import minmax_constraint
from ....registers import (
    asm_data_type as adt,
    asm_index_type as ait,
    adt_size
)

# Complete set of types
_ALL_DTS = [
    adt.FP64, adt.FP32, adt.FP16, adt.BF16,
    adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8,
    adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8
]

# x86 broadcasts, gathers, and scatters generally operate on dword/qword boundaries
_B32_64_DTS = [adt.FP64, adt.FP32, adt.SINT64, adt.SINT32, adt.UINT64, adt.UINT32]

_SIZE_AIT_MAP = {
    4 : ait.INT32,
    8 : ait.INT64
}

def make_avx_opmem_signatures(action: opmem_action,
                               is_avx512: bool = False, vbytes: int = 32) -> list[sig]:
    """
    Generate valid signatures for AVX/AVX2/AVX512 opmem operations.
    """
    sigs = []

    base_addr_mods = [set(), {mod.IOFFSET}, {mod.VOFFSET}]

    def add_sig(dt, *, mods, opd_mods = None):
        ops = {
            'adreg': osh(ot.REGISTER, orl.DATA, rt.VEC, dt),
            'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64)
        }
        structural_params = {}
        clobber_list = []

        if opd_mods is None:
            opd_mods = dict()

        for opd, omods in opd_mods.items():
            ops[opd].modifiers = omods

        maskrt = rt.MASK if is_avx512 else rt.VEC

        if mod.MASK in mods:
            ops['amreg'] = osh(ot.REGISTER, orl.MASK, maskrt, dt)

        # a 32-bit signed displacement: in bytes, or in vectors of the
        # register's size
        if mod.IOFFSET in mods:
            ops['ioffset'] = osh(ot.IMMEDIATE, orl.PARAM, None, None, value_constraints=[
                minmax_constraint(minval=-2**31, maxval=2**31 - 1)])
        if mod.VOFFSET in mods:
            ops['voffset'] = osh(ot.IMMEDIATE, orl.PARAM, None, None, value_constraints=[
                minmax_constraint(minval=-2**31 // vbytes, maxval=(2**31 - 1) // vbytes)])

        has_ilane = any(opd_mod.ILANE in mods for _,mods in opd_mods.items())

        if has_ilane:
            # Only 128 bits are addressable
            max_lane = (16 // adt_size(dt)) - 1
            ops['adreg_lane'] = osh(ot.IMMEDIATE, orl.PARAM, None, None,
                                    value_constraints=[
                                        minmax_constraint(minval=0, maxval=max_lane)])

        if mod.VINDEX in mods:
            # The index register is another vector
            ops['vidxreg'] = osh(ot.REGISTER, orl.ADDRESS, rt.VEC, dt)
            structural_params['it'] = _SIZE_AIT_MAP[adt_size(dt)]
            clobber_list.append('amreg')

        requires = set()
        if is_avx512 and mod.MASK in mods and mod.VINDEX not in mods and adt_size(dt) <= 2:
            # masked moves of bytes and words: vmovdqu8, vmovdqu16 (AVX512BW)
            requires.add("BW")
        if is_avx512 and any(opd_mod.ILANE in m for m in (opd_mods or {}).values()):
            # EVEX vpinsr/vpextr: bytes and words AVX512BW, dwords and
            # qwords AVX512DQ; the FP lanes (vmovsd, vmovhpd, vinsertps,
            # vextractps) AVX512F
            if adt_size(dt) <= 2:
                requires.add("BW")
            elif dt not in (adt.FP32, adt.FP64):
                requires.add("DQ")
        sigs.append(sig(modifiers=mods, operands=ops,
                        structural_params=structural_params,
                        clobber_list=clobber_list, requires=frozenset(requires)))


    # 1. Standard Load / Store
    # Applies to all data types and standard addressing
    for dt in _ALL_DTS:
        for addr_mod in base_addr_mods:
            add_sig(dt, mods=addr_mod)
            if is_avx512:
                # under a k mask: the lanes it holds read (the others zeroed)
                # or written - a last vector's first lanes
                add_sig(dt, mods=addr_mod | {mod.MASK})


    # 2. Broadcast (Loads only)
    # AVX vbroadcastss / vbroadcastsd only supports 32/64 bit boundaries
    if action == opmem_action.LOAD:
        for dt in _B32_64_DTS:
            for addr_mod in base_addr_mods:
                add_sig(dt, mods=addr_mod, opd_mods={'adreg':{opd_mod.BCAST}})


    # 3. Lane Loads / Stores
    # Usually restricted to XMM domain (16-byte) using vpinsr/vpextr, but mathematically
    # calculable for any size (throws NotImplementedError in implementation if missing).
    for dt in _ALL_DTS:
        for addr_mod in base_addr_mods:
            # lane loads don't preserve the lanes above the loaded one; a lane
            # store leaves the register as it is
            lane_mods = {opd_mod.ILANE, opd_mod.ZEROOTHERS} if action == opmem_action.LOAD \
                else {opd_mod.ILANE}
            add_sig(dt, mods=addr_mod, opd_mods={'adreg': lane_mods})


    # 4. Gather / Scatter (VINDEX)
    # Scatter is AVX-512 only. Both only support 32/64 bit data elements.
    # IOFFSET/VOFFSET cannot be combined with VINDEX in x86.
    if action == opmem_action.LOAD or (action == opmem_action.STORE and is_avx512):
        for dt in _B32_64_DTS:
            mods = {mod.VINDEX, mod.MASK}

            add_sig(dt, mods=mods)

    # an index register (x86's base + index x scale): beside every plain or
    # immediate-offset signature without lanes or vector indices - masked
    # ones (AVX-512's k) and broadcasts (BCAST, BLOCK) included -
    # unscaled, and scaled by 1, 2, 4 or 8 (SCALEOFFSET, scale_factor)
    for s0 in [s for s in sigs if not (s.modifiers - {mod.IOFFSET, mod.MASK})
               and 'vidxreg' not in s.operands
               and all(set(o.modifiers or ()) <= {opd_mod.BCAST, opd_mod.BLOCK}
                       for o in s.operands.values())]:
        offreg = osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64)
        sigs.append(sig(modifiers=set(s0.modifiers) | {mod.GOFFSET},
                        operands=dict(s0.operands, offreg=offreg),
                        structural_params=dict(s0.structural_params),
                        clobber_list=list(s0.clobber_list), requires=s0.requires))
        for scale in (1, 2, 4, 8):
            sigs.append(sig(modifiers=set(s0.modifiers) | {mod.GOFFSET, mod.SCALEOFFSET},
                            operands=dict(s0.operands, offreg=offreg),
                            structural_params=dict(s0.structural_params, scale_factor=scale),
                            clobber_list=list(s0.clobber_list), requires=s0.requires))
    def disp8(s, name):
        """
        disp8's range: one signed byte under VEX; under EVEX the multiples of
        the access's size N - the vector, an element broadcast - up to
        127*N (lanes are VEX: their xmm registers)
        """
        if "voffset" == name:
            return [minmax_constraint(minval=-128 if is_avx512 else -(128 // vbytes),
                                      maxval=127 if is_avx512 else 127 // vbytes)]
        adreg = s.operands.get("adreg")
        adreg_mods = adreg.modifiers if adreg is not None and adreg.modifiers else set()
        if not is_avx512 or opd_mod.ILANE in adreg_mods or adreg is None or adreg.dt is None:
            return [minmax_constraint(minval=-128, maxval=127)]
        n = adt_size(adreg.dt) if opd_mod.BCAST in adreg_mods else vbytes
        return [multiple_constraint(multiple=n), minmax_constraint(minval=-128 * n, maxval=127 * n)]

    return displacement_forms(sigs, disp8)
