"""Erase handwriting: rename every user-defined identifier to a generic token,
drop docstrings, and re-print the code from its syntax tree so formatting is
uniform. Builtins, imported names, attribute names and keyword-argument names
are kept, so the program still runs and still reads as the same algorithm.

Lever "erase handwriting": does the probe (and the word counter) survive when
surface identifiers and layout are gone?
"""

from __future__ import annotations

import ast
import builtins

BUILTIN_NAMES = set(dir(builtins))


class _Collector(ast.NodeVisitor):
    """Find names the code itself defines (assignment targets, function and class
    names, parameters, loop/comprehension/with/except targets, globals)."""

    def __init__(self) -> None:
        self.defined: list[str] = []
        self.imported: set[str] = set()

    def _add(self, name: str) -> None:
        if name not in self.defined:
            self.defined.append(name)

    def visit_Import(self, node):
        for a in node.names:
            self.imported.add((a.asname or a.name).split(".")[0])

    def visit_ImportFrom(self, node):
        for a in node.names:
            self.imported.add(a.asname or a.name)

    def visit_FunctionDef(self, node):
        self._add(node.name)
        for arg in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
            self._add(arg.arg)
        if node.args.vararg:
            self._add(node.args.vararg.arg)
        if node.args.kwarg:
            self._add(node.args.kwarg.arg)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        for arg in node.args.args:
            self._add(arg.arg)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self._add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._add(node.id)

    def visit_Global(self, node):
        for n in node.names:
            self._add(n)

    visit_Nonlocal = visit_Global

    def visit_ExceptHandler(self, node):
        if node.name:
            self._add(node.name)
        self.generic_visit(node)


class _Renamer(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]) -> None:
        self.m = mapping

    def visit_Name(self, node):
        if node.id in self.m:
            node.id = self.m[node.id]
        return node

    def visit_FunctionDef(self, node):
        node.name = self.m.get(node.name, node.name)
        for arg in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
            arg.arg = self.m.get(arg.arg, arg.arg)
        if node.args.vararg:
            node.args.vararg.arg = self.m.get(node.args.vararg.arg, node.args.vararg.arg)
        if node.args.kwarg:
            node.args.kwarg.arg = self.m.get(node.args.kwarg.arg, node.args.kwarg.arg)
        node.body = _drop_docstring(node.body)
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        for arg in node.args.args:
            arg.arg = self.m.get(arg.arg, arg.arg)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        node.name = self.m.get(node.name, node.name)
        node.body = _drop_docstring(node.body)
        self.generic_visit(node)
        return node

    def visit_Global(self, node):
        node.names = [self.m.get(n, n) for n in node.names]
        return node

    visit_Nonlocal = visit_Global

    def visit_ExceptHandler(self, node):
        if node.name:
            node.name = self.m.get(node.name, node.name)
        self.generic_visit(node)
        return node

    def visit_keyword(self, node):
        # keyword-argument names of calls to user functions are renamed too
        if node.arg in self.m:
            node.arg = self.m[node.arg]
        self.generic_visit(node)
        return node


def _drop_docstring(body):
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return body or [ast.Pass()]


def normalize_code(code: str) -> tuple[str, bool]:
    """Return (normalized_code, ok). On a parse failure returns the input
    unchanged with ok=False."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, False
    col = _Collector()
    col.visit(tree)
    mapping, fn_i, var_i = {}, 0, 0
    fn_names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    for name in col.defined:
        if name in col.imported or name in BUILTIN_NAMES or name.startswith("__"):
            continue
        if name in fn_names:
            mapping[name] = f"f{fn_i}"; fn_i += 1
        else:
            mapping[name] = f"v{var_i}"; var_i += 1
    tree.body = _drop_docstring(tree.body)
    tree = _Renamer(mapping).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), True
