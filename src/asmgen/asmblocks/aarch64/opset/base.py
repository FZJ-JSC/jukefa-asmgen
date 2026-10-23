"""
aarch64's set
"""
from ...op.opset import scalar_opset
from ....registers import asm_data_type as adt, adt_size, greg_base, freg_base
from .signatures import make_aarch64_opset_signatures

MASK64 = 2**64 - 1


def logical_immediate(value : int) -> bool:
    """
    Whether a 64-bit value is a bitmask immediate (orr's, so mov's): an
    element of 2 to 64 bits repeated, a rotated run of ones in it
    """
    v = value & MASK64
    if v in (0, MASK64):
        return False
    size = 64
    while size > 2:
        half = size // 2
        if (v & ((1 << half) - 1)) != ((v >> half) & ((1 << half) - 1)):
            break
        size = half
    mask = (1 << size) - 1
    elem = v & mask
    run = (1 << bin(elem).count("1")) - 1
    return any((((elem >> r) | (elem << (size - r))) & mask) == run for r in range(size))


def mov_encodes(value : int) -> bool:
    """
    Whether mov takes value as it is: movz's or movn's one 16-bit chunk, or
    a bitmask immediate
    """
    v = value & MASK64
    one_chunk = lambda x: sum(1 for i in range(4) if (x >> (16 * i)) & 0xFFFF) <= 1
    return one_chunk(v) or one_chunk(~v & MASK64) or logical_immediate(v)


class aarch64_opset(scalar_opset):
    """
    AArch64 set: mov x,#0; fmov from xzr or wzr
    """
    def make_signatures(self):
        return make_aarch64_opset_signatures()

    def zero_greg(self, *, dreg : greg_base) -> str:
        return self.asmwrap(f"mov {dreg},#0")

    def zero_freg(self, *, dreg : freg_base, dt : adt) -> str:
        return self.asmwrap(f"fmov {dreg},{'xzr' if 8 == adt_size(dt) else 'wzr'}")

    def one_freg(self, *, dreg : freg_base, dt : adt) -> str:
        return self.asmwrap(f"fmov {dreg},#1.0")

    def value_greg(self, *, dreg : greg_base, value : int) -> str:
        if mov_encodes(value):
            return self.asmwrap(f"mov {dreg},#{value}")
        # movz the lowest chunk that isn't zero, movk the others that aren't
        v = value & MASK64
        chunks = [(i, (v >> (16 * i)) & 0xFFFF) for i in range(4) if (v >> (16 * i)) & 0xFFFF]
        out = ""
        for n, (i, c) in enumerate(chunks):
            out += self.asmwrap(f"{'movz' if 0 == n else 'movk'} {dreg},#{c}" + (f", lsl #{16 * i}" if i else ""))
        return out
