"""
asmgen 0.6 answers questions instead of being probed, and keeps no shims
for older names: nothing in src/ asks hasattr, reads getattr with a default
or a literal name (an attribute read in disguise), writes attributes by
name (setattr, delattr), writes through __dict__ or vars(), or answers
attribute lookups in a module's __getattr__. getattr by a name computed at
run time - an operation by its action, a register constructor by its type
tag - is dispatch and stays.
"""
import ast
import os
import unittest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "asmgen")


def findings(tree, path):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name, args = node.func.id, node.args
            if name in ("hasattr", "setattr", "delattr"):
                out.append(f"{path}:{node.lineno}: {name}")
            elif name == "getattr" and len(args) > 2:
                out.append(f"{path}:{node.lineno}: getattr with a default")
            elif name == "getattr" and len(args) == 2 and isinstance(args[1], ast.Constant):
                out.append(f"{path}:{node.lineno}: getattr of a literal name ({args[1].value})")
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Subscript):
                    v = t.value
                    if isinstance(v, ast.Attribute) and v.attr == "__dict__":
                        out.append(f"{path}:{node.lineno}: a write through __dict__")
                    if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in ("vars", "globals"):
                        out.append(f"{path}:{node.lineno}: a write through {v.func.id}()")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
            out.append(f"{path}:{node.lineno}: a module __getattr__")
    return out


class test_no_probes(unittest.TestCase):
    def test_src(self):
        found = []
        for root, _, files in os.walk(SRC):
            for f in files:
                if f.endswith(".py"):
                    path = os.path.join(root, f)
                    with open(path, encoding="utf-8") as fh:
                        found += findings(ast.parse(fh.read()), os.path.relpath(path, SRC))
        self.assertEqual([], found, "\n".join(found))


if __name__ == "__main__":
    unittest.main()
