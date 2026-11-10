# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Facilities for generating variables/parameters in C++ code
"""
from enum import Enum, unique

headers_for_types = {
    "std::uint64_t" : "cstdint",
    "std::vector" : "vector"
}


@unique
class vio_type(Enum):
    """
    inline ASM operand type (input or output)
    """
    INPUT = 1
    OUTPUT = 2
    NONE = 4

class vargen:
    """
    Generates/declares+defines and tracks c++ variables for use as input/output parameters to
    inline asm, as well as headers their declarations depend on

    :param cppvars: Names of the C++ variables that were generated with this
    :type cppvars: list[str]
    :param cppvar_types: Dictionary mapping the variables onto their C++ types
    :type cppvar_types: dict[str,str]
    :param cppvar_inits: Dictionary mapping the variables onto C++ expressions
        that will initialize their respective inline ASM parameter
    :type cppvar_inits: dict[str,str]
    :param cppvar_cppinits: Dictionary mapping the variables onto C++ expressions
        that will initialize their value in C++ code
    :type cppvar_cppinits: dict[str,str]
    :param cppvar_vts: Dictionary mapping the variables onto their inline asm operand type
    :type cppvar_vts: dict[str,class:`asmgen.cppgen.declarations.vio_type`]
    :param required_headers: List of headers that must be included for the generated variables
    :type required_headers: list[str]
    :param extra_decl_init: String containing additional C++ code needed for 
        initializing the generated variables
    :type extra_decl_init: str
    """

    def __init__(self):
        """
        Constructor method
        """
        self.cppvars : list[str] = []
        self.cppvar_types : dict[str,str] = {}
        self.cppvar_inits : dict[str,str] = {}
        self.cppvar_cppinits : dict[str,str] = {}
        self.cppvar_vts : dict[str,vio_type] = {}
        self.required_headers : list[str] = []
        self.extra_decl_init : str = ""

    def new_var(self, *, cpp_type : str, name : str ="#auto#",
                cppinit : str = "",
                vt : vio_type =vio_type.OUTPUT) -> str:
        """
        Generate a new variable
        
        :param cpp_type: C++ type of the variable
        :type cpp_type: str
        :param name: Name of the variable
        :type name: str, optional
        :param vt: whether the variable will be an input or an output operand (output by default)
        :type vt: class:`asmgen.cppgen.declarations.vio_type`, optional
        :return: Name of the variable
        :rtype: str
        """
        if "#auto#" == name:
            name = f"tmpvar{len(self.cppvars)}"
        base_type = cpp_type
        if "<" in cpp_type:
            base_type = cpp_type.split("<")[0]
        if base_type in headers_for_types:
            self.required_headers.append(headers_for_types[base_type])
        self.cppvars.append(name)
        self.cppvar_types[name] = cpp_type
        self.cppvar_inits[name] = f"{name}"
        self.cppvar_cppinits[name] = cppinit
        self.cppvar_vts[name] = vt
        return name

    def new_vector(self, *, cpp_type : str, size : str | int=1,
                   fillwith : float=0.0,
                   name : str ="#auto#",
                   cppinit : str = "",
                   vt : vio_type =vio_type.OUTPUT) -> str:
        """
        Generate a new std::vector based variable
        
        :param cpp_type: C++ type of the variable
        :type cpp_type: str
        :param size: Either an integer size an elements or a C++ expression that will
            return the size
        :type size: Union[str,int]
        :param fillwith: An integer or FP value to initialize all vector elements with
        :type fillwith: Union[int,float]
        :param name: Name of the variable
        :type name: str, optional
        :param vt: whether the variable will be an input or an output operand (output by default)
        :type vt: class:`asmgen.cppgen.declarations.vio_type`, optional
        :return: Name of the variable
        :rtype: str
        """
        if "#auto#" == name:
            name = f"tmpvar{len(self.cppvars)}"
        base_type = cpp_type
        if base_type in headers_for_types:
            self.required_headers.append(headers_for_types[base_type])
        if "vector" not in self.required_headers:
            self.required_headers.append("vector")
        self.cppvars.append(name)
        self.cppvar_types[name] = f"std::vector<{cpp_type}>"
        self.cppvar_inits[name] = f"{name}ptr"
        self.cppvar_cppinits[name] = cppinit
        self.cppvar_vts[name] = vt

        self.extra_decl_init += f"{name}.resize({size});\n"
        self.extra_decl_init += f"std::fill({name}.begin(),{name}.end(),{fillwith});\n"
        self.extra_decl_init += f"void* {name}ptr = {name}.data();\n"

        return name

    def custom_var_init(self, *, name : str, init : str):
        """
        Sets a custom C++ initialization expression for the
        inline ASM parameter corresponding to a variable
        
        :param name: Name of the variable
        :type name: str, optional
        :param init: String containing a C++ expression for the initialization
        :type init: str
        """
        self.cppvar_inits[name] = init

    def custom_var_cppinit(self, *, name : str, cppinit : str):
        """
        Sets a custom C++ initialization expression for the variable
        
        :param name: Name of the variable
        :type name: str, optional
        :param init: String containing a C++ expression for the initialization
        :type init: str
        """
        self.cppvar_cppinits[name] = cppinit

    def get_declarations(self) -> str:
        """
        Returns the C++ code that declares all generated variables
        
        :return: String containing the C++ code containing all declarations
        :rtype: str
        """
        # false positive
        # pylint: disable=unnecessary-semicolon
        declarations = "\n".join(
                [f"{self.cppvar_types[name]} {name}{self.cppvar_cppinits[name]};"\
                        for name in self.cppvars])
        # pylint: enable=unnecessary-semicolon
        declarations += "\n"
        declarations += self.extra_decl_init
        return declarations

    def get_includes(self) -> str:
        """
        Returns the C++ code that includes all necessary headers
        
        :return: String containing the C++ code containing all declarations
        :rtype: str
        """
        includes = "\n".join([f"#include <{header}>" for header in self.required_headers])
        return includes

    def get_variables(self) -> list[tuple[str,str,vio_type]]:
        """
        Returns the generated C++ variables
        
        :return: list of tuples of (name, initialization expression, input/output type)
        :rtype: list[tuple[str,str,class:`asmgen.cppgen.declarations.vio_type`]]
        """
        return [(var,self.cppvar_inits[var],self.cppvar_vts[var]) for var in self.cppvars]

    def get_var_inits(self) -> dict[str,str]:
        """
        Returns the dictionary mapping variable names to their initialization expression
        
        :return: variable-initialization dict
        :rtype: dict[str,str]
        """
        return self.cppvar_inits

    def reset_variables(self):
        """
        Resets internal state, clearing all generated variables
        """
        self.cppvars = []
        self.cppvar_types = {}
        self.cppvar_inits = {}
        self.cppvar_vts = {}
        self.extra_decl_init = ""
