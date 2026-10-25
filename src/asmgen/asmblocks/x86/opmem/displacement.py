"""
x86 displacements (0.6, phase 5). An offset costs one byte (disp8, -128 to
127) or four (disp32, the full signed 32 bits) more than none; under EVEX
(AVX-512) the one byte is scaled by the access's size N (disp8*N), so there
the offsets one byte holds are the multiples of N from -128*N to 127*N.

Every load, store and prefetch form with an offset comes twice: the full 32
bits behind the encoding feature enable-disp32 (off by default) - first, so
it's taken where both are on - and disp8's range behind enable-disp8 (on by
default). With neither on there are no offset forms.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

from ...op.constraint import minmax_constraint
from ...op.signature import operation_signature

DISP8 = "enable-disp8"
DISP32 = "enable-disp32"
#: x86's encoding features and their defaults
ENCODING_FEATURES = {DISP8: True, DISP32: False}
#: the operands holding a displacement: bytes, or vectors (AVX)
DISPLACEMENTS = ("ioffset", "voffset")


def disp8_range(sig : operation_signature, name : str) -> list:
    """
    One signed byte: legacy and VEX encodings
    """
    return [minmax_constraint(minval=-128, maxval=127)]


def displacement_forms(sigs : list[operation_signature],
                       disp8 : Callable[[operation_signature, str], list] = disp8_range
                       ) -> list[operation_signature]:
    """
    Each signature with a displacement as two: the full signed 32 bits
    requiring enable-disp32, then disp8's range requiring enable-disp8

    :param disp8: the constraints of disp8's range, for a signature's
        displacement operand
    """
    out = []
    for s in sigs:
        names = [n for n in DISPLACEMENTS if n in s.operands]
        if not names:
            out.append(s)
            continue
        requires = frozenset(s.requires or ())
        out.append(dataclasses.replace(s, requires=requires | {DISP32}))
        operands = dict(s.operands)
        for n in names:
            operands[n] = dataclasses.replace(operands[n], value_constraints=disp8(s, n))
        out.append(dataclasses.replace(s, operands=operands, requires=requires | {DISP8}))
    return out
