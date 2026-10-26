"""
load and store with PARAM: the memory is an inline asm operand %[param],
there is no address register (0.6, phase 5). mov_param_to_greg,
mov_greg_to_param and load_pointer are those operations.
"""
import os
import shutil
import subprocess
import tempfile
import unittest

from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.avx import fma256, avx512
from asmgen.asmblocks.op.opmem import opmem_modifier as mod
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.op.opmem import opmem_modifier

PARAM = {mod.PARAM}

# generator, general purpose load and store of greg(5), FP64 load and FP32
# store of freg(2)
FORMS = [
    (rvv, '"ld t5, %[k]\\n\\t"', '"sd t5, %[k]\\n\\t"',
     '"fld f2, %[x]\\n\\t"', '"fsw f2, %[y]\\n\\t"'),
    (neon, '"ldr x5, %[k]\\n\\t"', '"str x5, %[k]\\n\\t"',
     '"ldr d2, %[x]\\n\\t"', '"str s2, %[y]\\n\\t"'),
    (sve, '"ldr x5, %[k]\\n\\t"', '"str x5, %[k]\\n\\t"',
     '"ldr d2, %[x]\\n\\t"', '"str s2, %[y]\\n\\t"'),
    (fma256, '"movq %[k], %%r13\\n\\t"', '"movq %%r13, %[k]\\n\\t"',
     '"vmovsd %[x], %%xmm2\\n\\t"', '"vmovss %%xmm2, %[y]\\n\\t"'),
    (avx512, '"movq %[k], %%r13\\n\\t"', '"movq %%r13, %[k]\\n\\t"',
     '"vmovsd %[x], %%xmm2\\n\\t"', '"vmovss %%xmm2, %[y]\\n\\t"'),
]


class test_param_forms(unittest.TestCase):
    def test_texts(self):
        for gen, gload, gstore, fload, fstore in FORMS:
            g = gen()
            with self.subTest(gen=gen.__name__):
                r = g.greg(5)
                self.assertEqual(gload, g.load(dregs=[r], dt=adt.UINT64, modifiers=PARAM, param="k").strip())
                self.assertEqual(gstore, g.store(dregs=[r], dt=adt.UINT64, modifiers=PARAM, param="k").strip())
                self.assertEqual(fload, g.load(dregs=[g.freg(2, adt.FP64)], dt=adt.FP64,
                                               modifiers=PARAM, param="x").strip())
                self.assertEqual(fstore, g.store(dregs=[g.freg(2, adt.FP32)], dt=adt.FP32,
                                                 modifiers=PARAM, param="y").strip())

    def test_rejected(self):
        for gen, *_ in FORMS:
            g = gen()
            with self.subTest(gen=gen.__name__):
                r = g.greg(5)
                with self.assertRaises(ValueError):    # neither an address register nor PARAM
                    g.load(dregs=[r], dt=adt.UINT64)
                with self.assertRaises(ValueError):    # PARAM has no offset forms
                    g.load(dregs=[r], dt=adt.UINT64, modifiers={mod.PARAM, mod.IOFFSET},
                           param="k", ioffset=8)


# generator, compiler, its flags, the register greg(5) is
COMPILES = [
    (rvv, "riscv64-linux-gnu-gcc", ["-march=rv64gc", "-mabi=lp64d"], "t5"),
    (neon, "aarch64-linux-gnu-gcc", [], "x5"),
    (fma256, "gcc", [], "r13"),
]


class test_param_in_inline_asm(unittest.TestCase):
    """
    The texts compile in an asm statement whose operands are memory ("m")
    """
    def test_compiles(self):
        for gen, cc, flags, clobber in COMPILES:
            with self.subTest(gen=gen.__name__):
                if not shutil.which(cc):
                    self.skipTest(f"{cc} not found")
                g = gen()
                r = g.greg(5)
                body = (g.load(dregs=[r], dt=adt.UINT64, modifiers={opmem_modifier.PARAM}, param="k") + g.store(dregs=[r], dt=adt.UINT64, modifiers={opmem_modifier.PARAM}, param="o")
                        + g.load(dregs=[r], dt=adt.UINT64, modifiers={opmem_modifier.PARAM}, param="k"))
                source = ("void f(long *in, long *out) {\n"
                          f"    __asm__ volatile({body}\n"
                          '        : [o] "=m"(*out)\n'
                          '        : [k] "m"(*in)\n'
                          f'        : "{clobber}");\n'
                          "}\n")
                with tempfile.TemporaryDirectory() as d:
                    c = os.path.join(d, "param.c")
                    with open(c, "w") as f:
                        f.write(source)
                    res = subprocess.run([cc, "-O2", *flags, "-c", c, "-o", os.path.join(d, "param.o")],
                                         capture_output=True, text=True)
                    self.assertEqual(0, res.returncode, res.stderr + source)


if __name__ == "__main__":
    unittest.main()
