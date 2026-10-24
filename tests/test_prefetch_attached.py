"""
gen.prefetch on every generator (0.6, phase 5): the base ISA's scalar
prefetch operation, plain and with IOFFSET. The legacy prefetch methods
(prefetch_l1_immoff, min_prefetch_offset, max_prefetch_offset) are gone:
nothing used them. Levels, write prefetches and non-temporal accesses come
with ukrgen 0.3.
"""
import unittest

from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.rvv071 import rvv071
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.sme import sme
from asmgen.asmblocks.avx import fma128, fma256, avx512
from asmgen.asmblocks.op.opmem import opmem_modifier as mod

# base ISA: prefetch of greg(5), plain and 64 bytes past it
FORMS = {
    "riscv64": ('"prefetch.r 0(t5)\\n\\t"', '"prefetch.r 64(t5)\\n\\t"'),
    "aarch64": ('"prfm PLDL1KEEP, [x5]\\n\\t"', '"prfm PLDL1KEEP, [x5, #64]\\n\\t"'),
    "x86": ('"prefetcht0 (%%r13)\\n\\t"', '"prefetcht0 64(%%r13)\\n\\t"'),
}
GENERATORS = [(rvv, "riscv64"), (rvv071, "riscv64"), (neon, "aarch64"), (sve, "aarch64"),
              (sme, "aarch64"), (fma128, "x86"), (fma256, "x86"), (avx512, "x86")]
LEGACY = ("prefetch_l1_immoff", "min_prefetch_offset", "max_prefetch_offset")


class test_prefetch_attached(unittest.TestCase):
    def test_every_generator(self):
        for gen, isa in GENERATORS:
            with self.subTest(gen=gen.__name__):
                g = gen()
                areg = g.greg(5)
                plain, offset = FORMS[isa]
                self.assertEqual(plain, g.prefetch(areg=areg).strip())
                self.assertEqual(offset, g.prefetch(areg=areg, modifiers={mod.IOFFSET}, ioffset=64).strip())

    def test_legacy_methods_gone(self):
        for gen, _ in GENERATORS:
            g = gen()
            for name in LEGACY:
                with self.subTest(gen=gen.__name__, method=name):
                    self.assertFalse(hasattr(g, name))


if __name__ == "__main__":
    unittest.main()
