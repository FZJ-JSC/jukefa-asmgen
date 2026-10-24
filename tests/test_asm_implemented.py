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
        'rvv' : ['add_greg_greg_shifted',
                 'sub_greg_greg_shifted',
                 'mask_all_true',
                 'vlen_to_greg',
                 'load_vector_bcast1_inc',
                 'load_vector_bcast1_immoff',
                 'load_vector_voff',
                 'load_vector_lane',
                 'load_vector_immstride',
                 'load_vector_immoff',
                 'store_vector_voff',
                 'store_vector_lane',
                 'store_vector_immstride',
                 'store_vector_immoff',
                 'load_tile',
                 'store_tile',
                 'zero_treg',
                 'treg',
                 'add_greg_voff',
                 'greg_to_voffs',
                 'fma_idx',
                 'fma_np_idx',
                 ],
        'rvv071' : ['add_greg_greg_shifted',
                 'sub_greg_greg_shifted',
                 'mask_all_true',
                 'vlen_to_greg',
                 'load_vector_bcast1_inc',
                    'load_vector_bcast1_immoff',
                    'load_vector_voff',
                    'load_vector_lane',
                    'load_vector_immstride',
                    'load_vector_immoff',
                    'store_vector_lane',
                    'store_vector_voff',
                    'store_vector_immstride',
                    'store_vector_immoff',
                    'load_tile',
                    'store_tile',
                    'zero_treg',
                    'treg',
                    'add_greg_voff',
                    'greg_to_voffs',
                    'fma_idx',
                    'fma_np_idx',
                    ],
        'sve' : ['load_vector_bcast1_inc',
                 'load_vector_immstride',
                 'load_vector_gregstride',
                 'load_vector_immoff',
                 'load_vector_lane',
                 'store_vector_lane',
                 'store_vector_immstride',
                 'store_vector_gregstride',
                 'store_vector_immoff',
                 'load_tile',
                 'store_tile',
                 'treg',
                 'zero_treg',
                 'fma_vf',
                 'fma_np_vf',
                 ],
        'sme' : ['load_vector_bcast1_inc',
                 'load_vector_immstride',
                 'load_vector_gregstride',
                 'load_vector_immoff',
                 'load_vector_lane',
                 'store_vector_lane',
                 'store_vector_immstride',
                 'store_vector_gregstride',
                 'store_vector_immoff',
                 'load_tile',
                 'store_tile',
                 'fma_vf',
                 'fma_np_vf',
                 ],
        'neon' : ['mask_all_true',
                 'vlen_to_greg',
                 'load_vector_bcast1_immoff',
                  'load_vector_immstride',
                  'load_vector_gregstride',
                  'load_vector_gather',
                  'store_vector_immstride',
                  'store_vector_gregstride',
                  'store_vector_scatter',
                  'load_tile',
                  'store_tile',
                  'zero_treg',
                  'treg',
                  'greg_to_voffs',
                  'fma_vf',
                  'fma_np_vf',
                  ],
        'fma128' : ['load_vector_gather',   # AVX2's gathers need a mask register
                   'sub_greg_greg_shifted',
                 'vlen_to_greg',
                 'load_vector_bcast1_inc',
                    'load_vector_immstride',
                    'load_vector_gregstride',
                    'load_vector_lane',
                    'store_vector_lane',
                    'store_vector_immstride',
                    'store_vector_gregstride',
                    'store_vector_scatter',
                    'load_tile',
                    'store_tile',
                    'zero_treg',
                    'treg',
                    'fma_idx',
                    'fma_np_idx',
                    'fma_vf',
                    'fma_np_vf',
                    ],
        'fma256' : ['load_vector_gather',   # AVX2's gathers need a mask register
                   'sub_greg_greg_shifted',
                 'vlen_to_greg',
                 'load_vector_bcast1_inc',
                    'load_vector_immstride',
                    'load_vector_gregstride',
                    'load_vector_lane',
                    'store_vector_lane',
                    'store_vector_immstride',
                    'store_vector_gregstride',
                    'store_vector_scatter',
                    'load_tile',
                    'store_tile',
                    'zero_treg',
                    'treg',
                    'fma_idx',
                    'fma_np_idx',
                    'fma_vf',
                    'fma_np_vf',
                    ],
        'avx512' : ['sub_greg_greg_shifted',
                 'vlen_to_greg',
                 'load_vector_bcast1_inc',
                    'load_vector_immstride',
                    'load_vector_gregstride',
                    'load_vector_lane',
                    'store_vector_lane',
                    'store_vector_immstride',
                    'store_vector_gregstride',
                    'load_tile',
                    'store_tile',
                    'zero_treg',
                    'treg',
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
        ['indexable_elements', {'dt' : lambda gen : adt.DOUBLE} ],
        ['max_vregs', None],
        ['max_gregs', None],
        ['max_fregs', None],
        ['max_tregs', {'dt' : lambda gen : adt.DOUBLE} ],
        ['min_bcast_immoff', {'dt' : lambda gen : adt.DOUBLE} ],
        ['min_fload_immoff', {'dt' : lambda gen : adt.DOUBLE} ],
        ['max_bcast_immoff', {'dt' : lambda gen : adt.DOUBLE} ],
        ['max_fload_immoff', {'dt' : lambda gen : adt.DOUBLE} ],
        ['min_load_voff', None],
        ['max_load_voff', None],
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
        ['min_load_immoff', {'dt' : lambda gen : adt.DOUBLE} ],
        ['max_load_immoff', {'dt' : lambda gen : adt.DOUBLE} ],
        ['min_load_immoff', {'dt' : lambda gen : adt.SINGLE} ],
        ['max_load_immoff', {'dt' : lambda gen : adt.SINGLE} ],
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
        ['simd_size_to_greg', {'reg' : lambda gen : gen.greg(0),
                               'dt' : lambda gen : adt.SINGLE} ],
        ['load_greg', {'areg' : lambda gen : gen.greg(0),
                       'offset' : lambda gen : 4,
                       'dst' : lambda gen : gen.greg(1)}],
        ['store_greg', {'areg' : lambda gen : gen.greg(0),
                        'offset' : lambda gen : 4,
                        'src' : lambda gen : gen.greg(1)}],
        ['zero_greg', {'greg' : lambda gen : gen.greg(0)}],
        ['mov_greg', {'src' : lambda gen : gen.greg(0),
                      'dst' : lambda gen : gen.greg(1)}],
        ['mov_greg_to_param', {'src' : lambda gen : gen.greg(0),
                               'param' : lambda gen : 'someparam'}],
        ['mov_param_to_greg', {'dst' : lambda gen : gen.greg(0),
                               'param' : lambda gen : 'someparam'}],
        ['mov_param_to_greg_shift', {'dst' : lambda gen : gen.greg(0),
                                     'param' : lambda gen : 'someparam',
                                     'bit_count' : lambda gen : 2}],
        ['mov_greg_imm', {'reg' : lambda gen : gen.greg(0),
                          'imm' : lambda gen : 40}],
        ['add_greg_imm', {'reg' : lambda gen : gen.greg(0),
                          'imm' : lambda gen : 40}],
        ['mul_greg_greg', {'dst' : lambda gen : gen.greg(0),
                           'reg1' : lambda gen : gen.greg(1),
                           'reg2' : lambda gen : gen.greg(2)}],
        ['add_greg_greg', {'dst' : lambda gen : gen.greg(0),
                           'reg1' : lambda gen : gen.greg(1),
                           'reg2' : lambda gen : gen.greg(2)}],
        ['sub_greg_greg', {'dst' : lambda gen : gen.greg(0),
                           'reg1' : lambda gen : gen.greg(1),
                           'reg2' : lambda gen : gen.greg(2)}],
        ['add_greg_greg_shifted', {'dst' : lambda gen : gen.greg(0),
                                   'reg1' : lambda gen : gen.greg(1),
                                   'reg2' : lambda gen : gen.greg(2),
                                   'shift' : lambda gen : 1}],
        ['sub_greg_greg_shifted', {'dst' : lambda gen : gen.greg(0),
                                   'reg1' : lambda gen : gen.greg(1),
                                   'reg2' : lambda gen : gen.greg(2),
                                   'shift' : lambda gen : 1}],
        ['mask_all_true', {'reg' : lambda gen : gen.mreg(1) if hasattr(gen, 'mreg') else gen.vreg(0),
                           'dt' : lambda gen : adt.FP64}],
        ['vlen_to_greg', {'reg' : lambda gen : gen.greg(0),
                          'dt' : lambda gen : adt.FP64}],
        ['mov_greg_to_freg', {'src' : lambda gen : gen.greg(0),
                              'dst' : lambda gen : gen.freg(1, adt.FP64),
                              'dt' : lambda gen : adt.FP64}],
        ['ret', {}],
        ['add_greg_voff', {'reg' : lambda gen : gen.greg(0),
                           'offset' : lambda gen : 40,
                           'dt' : lambda gen : adt.DOUBLE}],
        ['add_greg_voff', {'reg' : lambda gen : gen.greg(0),
                           'offset' : lambda gen : 40,
                           'dt' : lambda gen : adt.SINGLE}],
        ['mul_greg_imm', {'dst' : lambda gen : gen.greg(0),
                          'src' : lambda gen : gen.greg(1),
                          'factor' : lambda gen : 40}],
        ['shift_greg_left', {'reg' : lambda gen : gen.greg(0),
                             'bit_count' : lambda gen : 4}],
        ['shift_greg_right', {'reg' : lambda gen : gen.greg(0),
                             'bit_count' : lambda gen : 4}],
        ['greg_to_voffs', {'streg' : lambda gen : gen.greg(0),
                           'vreg' : lambda gen : gen.vreg(1),
                           'dt' : lambda gen : adt.SINGLE} ],
        ['kiterkleft', {'kreg' : lambda gen : gen.greg(0),
                        'kleftreg' : lambda gen : gen.greg(1),
                        'tmpreg' : lambda gen : gen.greg(2),
                        'unroll' : lambda gen : 4} ],
        ['mov_freg', {'src' : lambda gen : gen.freg(0, adt.SINGLE),
                      'dst' : lambda gen : gen.freg(1, adt.SINGLE),
                      'dt' : lambda gen : adt.SINGLE} ],
        ['zero_vreg', {'vreg' : lambda gen : gen.vreg(0),
                       'dt' : lambda gen : adt.DOUBLE}],
        ['zero_vreg', {'vreg' : lambda gen : gen.vreg(0),
                       'dt' : lambda gen : adt.SINGLE}],
        ['zero_freg', {'freg' : lambda gen : gen.freg(0, adt.SINGLE),
                       'dt' : lambda gen : adt.SINGLE}],
        ['zero_treg', {'treg' : lambda gen : gen.treg(0, adt.SINGLE),
                       'dt' : lambda gen : adt.SINGLE}],
        ['load_pointer', {'areg' : lambda gen : gen.greg(0),
                          'name' : lambda gen : 'someparam'}],
        ['load_tile', {'areg' : lambda gen : gen.greg(0),
                       'treg' : lambda gen : gen.treg(0),
                       'dt' :  lambda gen : adt.DOUBLE}],
        ['load_vector', {'areg' : lambda gen : gen.greg(0),
                         'vreg' : lambda gen : gen.vreg(0),
                         'dt' :  lambda gen : adt.DOUBLE}],
        ['load_vector', {'areg' : lambda gen : gen.greg(0),
                         'vreg' : lambda gen : gen.vreg(0),
                         'dt' :  lambda gen : adt.DOUBLE}],
        ['fill_vector', {'sreg' : lambda gen : gen.freg(0, dt=adt.DOUBLE),
                         'vreg' : lambda gen : gen.vreg(1),
                         'dt' : lambda gen : adt.DOUBLE}],
        ['load_vector_lane', {'areg' : lambda gen : gen.greg(0),
                              'vreg' : lambda gen : gen.vreg(0),
                              'lane' : lambda gen : 1,
                              'dt' :  lambda gen : adt.DOUBLE}],
        ['load_vector_voff', {'areg' : lambda gen : gen.greg(0),
                              'voffset' : lambda gen : 4,
                              'vreg' : lambda gen : gen.vreg(0),
                              'dt' :  lambda gen : adt.SINGLE}],
        ['load_vector_immoff', {'areg' : lambda gen : gen.greg(0),
                                'offset' : lambda gen : 4,
                                'vreg' : lambda gen : gen.vreg(0),
                                'dt' :  lambda gen : adt.SINGLE}],
        ['load_freg', {'areg' : lambda gen : gen.greg(0),
                       'offset' : lambda gen : 4,
                       'dst' : lambda gen : gen.freg(0, adt.DOUBLE),
                       'dt' :  lambda gen : adt.DOUBLE}],
        ['load_scalar_immoff', {'areg' : lambda gen : gen.greg(0),
                                'offset' : lambda gen : 4,
                                'freg' : lambda gen : gen.freg(0, adt.DOUBLE),
                                'dt' :  lambda gen : adt.DOUBLE}],
        ['load_vector_bcast1', {'areg' : lambda gen : gen.greg(0),
                               'vreg' : lambda gen : gen.vreg(0),
                               'dt' :  lambda gen : adt.DOUBLE}],
        ['load_vector_bcast1', {'areg' : lambda gen : gen.greg(0),
                               'vreg' : lambda gen : gen.vreg(0),
                               'dt' :  lambda gen : adt.SINGLE}],
        ['load_vector_bcast1_immoff', {'areg' : lambda gen : gen.greg(0),
                                    'offset' : lambda gen : 8,
                                    'vreg' : lambda gen : gen.vreg(0),
                                    'dt' :  lambda gen : adt.DOUBLE}],
        ['load_vector_bcast1_immoff', {'areg' : lambda gen : gen.greg(0),
                                    'offset' : lambda gen : 4,
                                    'vreg' : lambda gen : gen.vreg(0),
                                    'dt' :  lambda gen : adt.SINGLE}],
        ['load_vector_bcast1_inc', {'areg' : lambda gen : gen.greg(0),
                                   'offset' : lambda gen : 8,
                                   'vreg' : lambda gen : gen.vreg(0),
                                   'dt' :  lambda gen : adt.DOUBLE}],
        ['load_vector_bcast1_inc', {'areg' : lambda gen : gen.greg(0),
                                   'offset' : lambda gen : 4,
                                   'vreg' : lambda gen : gen.vreg(0),
                                   'dt' :  lambda gen : adt.SINGLE}],
        ['load_vector_immstride', {'areg' : lambda gen : gen.greg(0),
                                   'byte_stride' : lambda gen : 4,
                                   'vreg' : lambda gen : gen.vreg(0),
                                   'dt' :  lambda gen : adt.SINGLE}],
        ['load_vector_gregstride', {'areg' : lambda gen : gen.greg(0),
                                    'sreg' : lambda gen : gen.greg(1),
                                    'vreg' : lambda gen : gen.vreg(0),
                                    'dt' :  lambda gen : adt.SINGLE}],
        ['load_vector_gather', {'areg' : lambda gen : gen.greg(0),
                                'offvreg' : lambda gen : gen.vreg(1),
                                'vreg' : lambda gen : gen.vreg(0),
                                'dt' :  lambda gen : adt.SINGLE,
                                'it' :  lambda gen : ait.INT32}],
        ['store_freg', {'areg' : lambda gen : gen.greg(0),
                        'offset' : lambda gen : 4,
                        'src' : lambda gen : gen.freg(0, adt.DOUBLE),
                        'dt' :  lambda gen : adt.DOUBLE}],
        ['store_scalar_immoff', {'areg' : lambda gen : gen.greg(0),
                                 'offset' : lambda gen : 4,
                                 'freg' : lambda gen : gen.freg(0, adt.DOUBLE),
                                 'dt' :  lambda gen : adt.DOUBLE}],
        ['store_tile', {'areg' : lambda gen : gen.greg(0),
                       'treg' : lambda gen : gen.treg(0),
                       'dt' :  lambda gen : adt.DOUBLE}],
        ['store_vector', {'areg' : lambda gen : gen.greg(0),
                          'vreg' : lambda gen : gen.vreg(0),
                          'dt' :  lambda gen : adt.DOUBLE}],
        ['store_vector_lane', {'areg' : lambda gen : gen.greg(0),
                               'vreg' : lambda gen : gen.vreg(0),
                               'lane' : lambda gen : 1,
                               'dt' :  lambda gen : adt.DOUBLE}],
        ['store_vector_immoff', {'areg' : lambda gen : gen.greg(0),
                                 'offset' : lambda gen : 4,
                                 'vreg' : lambda gen : gen.vreg(0),
                                 'dt' :  lambda gen : adt.SINGLE}],
        ['store_vector_voff', {'areg' : lambda gen : gen.greg(0),
                               'voffset' : lambda gen : 4,
                               'vreg' : lambda gen : gen.vreg(0),
                               'dt' :  lambda gen : adt.SINGLE}],
        ['store_vector_immstride', {'areg' : lambda gen : gen.greg(0),
                                    'byte_stride' : lambda gen : 4,
                                    'vreg' : lambda gen : gen.vreg(0),
                                    'dt' :  lambda gen : adt.SINGLE}],
        ['store_vector_gregstride', {'areg' : lambda gen : gen.greg(0),
                                     'sreg' : lambda gen : gen.greg(1),
                                     'vreg' : lambda gen : gen.vreg(0),
                                     'dt' :  lambda gen : adt.SINGLE}],
        ['store_vector_scatter', {'areg' : lambda gen : gen.greg(0),
                                  'offvreg' : lambda gen : gen.vreg(1),
                                  'vreg' : lambda gen : gen.vreg(0),
                                  'dt' :  lambda gen : adt.SINGLE,
                                  'it' :  lambda gen : ait.INT32}],
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
            '_scalar_memory',       # the scalar loads' and stores', tested through them
            '_param_memory',        # the parameter loads' and stores', tested through them
            '_vector_memory',       # the old vector loads' and stores', tested through them
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
