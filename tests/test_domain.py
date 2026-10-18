"""
Values an operand may take under its constraints; register allocation
priorities
"""

import unittest

from asmgen.asmblocks.op.constraint import (
    ArgumentDependencyError, minmax_constraint, oneof_constraint, multiple_constraint,
    either_constraint, otherplusn_constraint, differentval_constraint,
)
from asmgen.asmblocks.op.domain import constraint_domain, unbounded_domain_error
from asmgen.registers import reg_tracker


class test_domain(unittest.TestCase):

    def test_finite_first(self):
        # a multiple filters what a range bounds, in whichever order given
        d = constraint_domain([multiple_constraint(multiple=8), minmax_constraint(minval=-16, maxval=16)])
        self.assertEqual([-16, -8, 0, 8, 16], d.values)

    def test_either_union(self):
        d = constraint_domain([either_constraint(options=[[minmax_constraint(minval=-2, maxval=-1)],
                                                           [oneof_constraint(valset={4, 8})]])])
        self.assertEqual([-2, -1, 4, 8], d.values)

    def test_unbounded(self):
        with self.assertRaises(unbounded_domain_error):
            constraint_domain([multiple_constraint(multiple=8)])
        self.assertEqual([0, 8], constraint_domain([multiple_constraint(multiple=8)],
                                                   universe=range(16)).values)
        self.assertIsNone(constraint_domain([]).values)

    def test_dependencies(self):
        c = otherplusn_constraint(other="a", offset=2)
        with self.assertRaises(ArgumentDependencyError):
            constraint_domain([c])
        self.assertEqual([7], constraint_domain([c], context={"a": 5}).values)
        d = constraint_domain([differentval_constraint(other="a")], context={"a": 1}, universe=range(4))
        self.assertEqual([0, 2, 3], d.values)
        # the constraint's own check raises the dependency too, not a KeyError
        with self.assertRaises(ArgumentDependencyError):
            c("b", set(), 7, {})

    def test_unknown_called(self):
        class odd(minmax_constraint):
            pass
        class opaque(oneof_constraint.__mro__[1]):
            def validate(self, name, val, context, params):
                if val % 2:
                    raise ValueError("odd")
        d = constraint_domain([opaque(), minmax_constraint(minval=0, maxval=5)])
        self.assertEqual([0, 2, 4], d.values)


class test_priorities(unittest.TestCase):

    def test_highest_first(self):
        rt = reg_tracker([("greg", 8)])
        for n in range(4):
            rt.set_priority("greg", n, -1)
        rt.set_priority("greg", 6, 1)
        self.assertEqual([6, 4, 5, 7, 0, 1, 2, 3], [rt.reserve_any_reg("greg") for _ in range(8)])


if __name__ == "__main__":
    unittest.main()


class test_ranges(unittest.TestCase):
    """
    Large domains stay ranges: a 32-bit displacement is never listed
    """

    def test_large_symbolic(self):
        d = constraint_domain([minmax_constraint(minval=-2**31, maxval=2**31 - 1)])
        self.assertEqual(2**32, d.size)
        self.assertIn(-5, d)
        self.assertNotIn(2**31, d)
        with self.assertRaises(unbounded_domain_error):
            d.values

    def test_scaled_offsets(self):
        # AArch64's ldr d: scaled unsigned, or ldur's unscaled signed
        d = constraint_domain([either_constraint(options=[
            [minmax_constraint(minval=0, maxval=4095 * 8), multiple_constraint(multiple=8)],
            [minmax_constraint(minval=-256, maxval=255)]])])
        for ok in (-256, -1, 0, 7, 255, 256, 32760):
            self.assertIn(ok, d)
        for bad in (-257, 257, 32761, 32768):
            self.assertNotIn(bad, d)


class test_every_immediate_constrained(unittest.TestCase):
    """
    Every immediate operand of every signature has constraints, and none
    leaves its operand without a value
    """

    def test_all(self):
        from asmgen.asmblocks.op.operand import operand_type as ot
        from asmgen.asmblocks.neon import neon
        from asmgen.asmblocks.sve import sve
        from asmgen.asmblocks.rvv import rvv
        from asmgen.asmblocks.sme import sme
        from asmgen.asmblocks.avx.avx2 import fma256
        from asmgen.asmblocks.avx.avx512 import avx512
        for G in (neon, sve, sme, rvv, fma256, avx512):
            g = G()
            for opn in ("load", "store", "move"):
                for s in getattr(g, opn).get_signatures():
                    for name, o in s.operands.items():
                        if ot.IMMEDIATE != o.otype:
                            continue
                        self.assertTrue(o.value_constraints, f"{G.__name__}.{opn}.{name}")
                        try:
                            d = constraint_domain(o.value_constraints, name=name,
                                                  modifiers=set(s.modifiers))
                        except ArgumentDependencyError:
                            continue
                        self.assertNotEqual(0, d.size, f"{G.__name__}.{opn}.{name}")
