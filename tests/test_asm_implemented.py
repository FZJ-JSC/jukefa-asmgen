# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Test the asmgen implementations for different ISAs on whether they implement
required methods
"""
import unittest
import inspect

from parameterized import parameterized, parameterized_class

from asmgen.asmblocks.noarch import asmgen,comparison
from asmgen.registers import asm_data_type as adt
from asmgen.registers import asm_index_type as ait
from asmgen.registers import reg_tracker

from asmgen.asmblocks.avx import fma128,fma256,avx512
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.sme import sme
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.rvv071 import rvv071

from .testcase import testcase


allowed_not_implemented = {
        'rvv' : ['simd_size',   # vector-length-agnostic: known at run time only
                   'treg',
                 'fma_idx',
                 'fma_np_idx',
                 ],
        'rvv071' : ['simd_size',   # vector-length-agnostic: known at run time only
                   'treg',
                    'fma_idx',
                    'fma_np_idx',
                    ],
        'sve' : ['simd_size',   # vector-length-agnostic: known at run time only
                   'treg',
                 'fma_vf',
                 'fma_np_vf',
                 ],
        'sme' : ['simd_size',   # vector-length-agnostic: known at run time only
                   'fma_vf',
                 'fma_np_vf',
                 ],
        'neon' : ['treg',
                  'fma_vf',
                  'fma_np_vf',
                  ],
        'fma128' : ['treg',
                    'fma_idx',
                    'fma_np_idx',
                    'fma_vf',
                    'fma_np_vf',
                    ],
        'fma256' : ['treg',
                    'fma_idx',
                    'fma_np_idx',
                    'fma_vf',
                    'fma_np_vf',
                    ],
        'avx512' : ['treg',
                    'fma_idx',
                    'fma_np_idx',
                    'fma_vf',
                    'fma_np_vf',
                    ]
}

rts = dict()
def get_rt(gen : asmgen, name: str) -> reg_tracker:
    """
    get register tracker for a generator (or build it if it doesn't exist)
    """
    if name not in rts:
        rts[name] = reg_tracker(
                reg_type_init_list=[
                    ('greg', gen.max_gregs()),
                    ('freg', gen.max_fregs()),
                    ('vreg', gen.max_vregs()),
                    ('treg', gen.max_tregs(adt.FP64)),
                    ]
                )
    return rts[name]


function_tests = [
        ['simd_size', None],
        ['get_parameters', None],
        ['get_param_value', {'name' : lambda gen : 'dummy'}],
        ['c_simd_size_function', None],
        ['is_vla', None],
        ['supported_on_host', None],
        ['supportedby_cpuinfo', {'cpuinfo' : lambda gen : 'invalid_isa'}],
        ['max_vregs', None],
        ['max_gregs', None],
        ['max_fregs', None],
        ['max_tregs', {'dt' : lambda gen : adt.DOUBLE} ],
        ['create_callconv', {'name' : lambda gen : "default"}],
        ['isaclear', None],
        ['isadata', None],
        ['isaclear', None],
        ['isaquirks', {
            'rt' : lambda gen : get_rt(gen=gen, name=gen.__class__),
            'dt' : lambda gen : adt.SINGLE} ],
        ['isaendquirks', {
            'rt' : lambda gen : get_rt(gen=gen, name=gen.__class__),
            'dt' : lambda gen : adt.SINGLE} ],
        ['treg', { 'reg_idx' : lambda gen :  0, 'dt' : lambda gen : adt.DOUBLE} ],
        ['freg', { 'reg_idx' : lambda gen :  0, 'dt' : lambda gen : adt.DOUBLE} ],
        ['greg', { 'reg_idx' : lambda gen :  0} ],
        ['vreg', { 'reg_idx' : lambda gen :  0} ],
        ['cb', {'reg1' : lambda gen: gen.greg(0),
                'reg2' : lambda gen: gen.greg(1),
                'cmp' : lambda gen: comparison.NE,
                'label' : lambda gen : 'notequal'}],
        ['jump', {'label' : lambda gen : 'someloop'}],
        ['jzero', {'reg' : lambda gen: gen.greg(0),
                    'label' : lambda gen : 'someloop'}],
        ['jfzero', {'freg1' : lambda gen: gen.freg(0, adt.SINGLE),
                    'freg2' : lambda gen: gen.freg(1, adt.SINGLE),
                    'greg' : lambda gen: gen.greg(0),
                    'label' : lambda gen : 'someloop',
                    'dt' : lambda gen : adt.SINGLE} ],
        ['jvzero', {'vreg1' : lambda gen: gen.vreg(0),
                    'freg' : lambda gen: gen.freg(2, adt.SINGLE),
                    'vreg2' : lambda gen: gen.vreg(1),
                    'greg' : lambda gen: gen.greg(0),
                    'label' : lambda gen : 'someloop',
                    'dt' : lambda gen : adt.SINGLE} ],
        ['loopbegin', {'reg' : lambda gen: gen.greg(0),
                       'label' : lambda gen : 'someloop'}],
        ['loopbegin_nz', {'reg' : lambda gen: gen.greg(0),
                         'label' : lambda gen : 'someloop',
                         'labelskip' : lambda gen : 'someloop_nz'}],
        ['loopend', {'reg' : lambda gen: gen.greg(0),
                     'label' : lambda gen : 'someloop'}],
        ['label', {'label' : lambda gen : "label"}],
        ['labelstr', {'label' : lambda gen : "label"}],
        ['ret', {}],
        ['nop', {}],
        ['align', {'bytes' : lambda gen : 64}],
        ['asmfile_annotations', {}],
        ['enabled_features', {}],
        ['compiler_flags', {}],
        ['kiterkleft', {'kreg' : lambda gen : gen.greg(0),
                        'kleftreg' : lambda gen : gen.greg(1),
                        'tmpreg' : lambda gen : gen.greg(2),
                        'unroll' : lambda gen : 4} ],
    ]

@parameterized_class([
    {"name": "fma128", "gen": fma128(), "allowed": allowed_not_implemented['fma128']},
    {"name": "fma256", "gen": fma256(), "allowed": allowed_not_implemented['fma256']},
    {"name": "avx512", "gen": avx512(), "allowed": allowed_not_implemented['avx512']},
    {"name": "neon", "gen": neon(), "allowed": allowed_not_implemented['neon']},
    {"name": "sve", "gen": sve(), "allowed": allowed_not_implemented['sve']},
    {"name": "sme", "gen": sme(), "allowed": allowed_not_implemented['sme']},
    {"name": "rvv", "gen": rvv(), "allowed": allowed_not_implemented['rvv']},
    {"name": "rvv071", "gen": rvv071(), "allowed": allowed_not_implemented['rvv071']},
])
class asm_implementation_test(unittest.TestCase, testcase):
    """
    Parameterized test that tests implementation of multiple methods in multiple generators 
    """
    @parameterized.expand(function_tests)
    def test_asmgen_method_implemented(self, name, args):
        """
        Tests whether a method is implemented in a generator by calling it with
        reasonable arguments, resulting in a fail if the call fails
        """
        if name in self.allowed:
            self.skipTest(f"Generator allowed to not implement {name}")

        copy_args = {}
        if args:
            for k in args.keys():
                copy_args[k] = args[k](self.gen)

        method = getattr(self.gen, name)
        if callable(method):
            method(**copy_args)

class asm_test_implementation_tests(unittest.TestCase):
    """
    Tests the implementation tester for completeness
    """
    def test_all_funcs_tested(self):
        """
        Tests whether all methods that the base asmgen class declares
        are being tested
        """

        allowed_not_tested= [
            '__init__',
            '_c_simd_size_asm',     # c_simd_size_function's, tested through it
            '_encoding_features',   # through get_param_value and set_parameter
            '_gate_features',       # through set_features and set_parameter
            '__annotate_func__',
            'operands',
            'set_output_inline',
            'set_parameter',
            'asmwrap',
            # configuring the generator, not generating instructions
            'available_features',
            'get_reg_tracker',
            'isa_setup',
            'reserved_registers',
            'set_features',
            'shared_registers',
        ]

        existing_methods = inspect.getmembers(asmgen,
                                              predicate=inspect.isfunction)
        existing_methods = [method[0] for method in existing_methods if \
                            method[0] not in allowed_not_tested]
        tested_methods = [method[0] for method in function_tests]

        untested = 0
        for method in existing_methods:
            if method not in tested_methods:
                untested += 1
                print(f"Implementation test for asmgen method {method} missing")
        self.assertEqual(untested, 0)
