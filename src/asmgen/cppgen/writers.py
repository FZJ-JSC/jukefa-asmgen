# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Functions that write C++ code from templates and parameters
"""
from mako.template import Template

import asmgen.cppgen.templates as cpptpl


def write_test_asmblock_func(name : str, asmblock : str, tparams : dict[str,str]) -> str:
    """
    Writes a function that tests an ASM block

    :param name: function name
    :param asmblock: String containing the inline asm block
    :param tparams: Dictionary containing requires template parameters
    :return: String containing the source code of the function
    """
    template = cpptpl.TEST_FUNC
    func_source = Template(template).render(
            asmblock=asmblock,
            function_name=name,
            function_params=tparams["function_params"],
            header=tparams["header"],
            prepare=tparams["prepare"],
            checkfun_definition=tparams["checkfun_definition"],
            check=tparams["check"],
            analyze=tparams["analyze"])
    return func_source+"\n\n"

def write_test_func_declaration(name : str, tparams : dict[str,str]) -> str:
    """
    Writes only the declaration of the function that tests an ASM block

    :param name: function name
    :param tparams: Dictionary containing requires template parameters
    :return: String containing the source code of the function declaration
    """
    template = cpptpl.TEST_FUNC_DECLARATION
    func_source = Template(template).render(
            function_name=name,
            function_params=tparams["function_params"])
    return func_source+"\n\n"

def write_asmblock_func(name : str, asmblock : str, tparams : dict[str,str]) -> str:
    """
    Writes a function that executes an ASM block

    :param name: function name
    :param asmblock: String containing the inline asm block
    :param tparams: Dictionary containing requires template parameters
    :return: String containing the source code of the function
    """
    template = cpptpl.SIMPLE_FUNC
    func_source = Template(template).render(
            asmblock=asmblock,
            function_name=name,
            function_params=tparams["function_params"],
            prepare=tparams["prepare"])
    return func_source+"\n\n"

def write_standalone_asmblock_test(asmblock : str, tparams : dict[str,str]) -> str:
    """
    Writes a standalone test for an ASM block

    :param asmblock: String containing the inline asm block
    :param tparams: Dictionary containing requires template parameters
    :return: String containing the source code of the test
    """
    template = cpptpl.SIMPLE_TEST
    func_source = Template(template).render(
            asmblock=asmblock,
            header=tparams["header"],
            checkfun_definition=tparams["checkfun_definition"],
            tmp_decl=tparams["tmp_decl"],
            tmp_assign=tparams["tmp_assign"],
            inparam_definition=tparams["inparam_definition"],
            outparam_definition=tparams["outparam_definition"],
            check=tparams["check"],
            analyze=tparams["analyze"])
    return func_source+"\n\n"
