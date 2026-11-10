# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Test the C++ source code writers
"""

import unittest

from asmgen.cppgen.writers import write_asmblock_func


class cpp_writers_test(unittest.TestCase):
    """
    Test case for C++ source code writers
    """
    def test_asmblock_func_no_tmpl_left(self):
        """
        Tests for no remaining template parameters in generated C++ code
        """
        cppsource = write_asmblock_func("fake_func", "",
                                        {
                                            "function_params":"",
                                            "prepare":"",
                                            })
        self.assertNotIn("${", cppsource)
