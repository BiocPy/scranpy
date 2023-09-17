#!/usr/bin/python3

# This script generates a dry-run version of the live_analyze function in
# src/scranpy/analyze.py. The idea is to allow users to create a dry-run
# of all the individual commands that they can copy-paste into their
# scripts, if they want to customize the analysis but they don't want
# to actually deal with learning all of the individual steps.
#
# The meta-parsing rules are as follows:
# - We only consider code after the initialization of 'results'.
# - Control flow (if, for, raise) statements are evaluated immediately,
#   under the assumption that they only use 'options'.
# - Everything else is stringified.

import ast
import sys
stuff = open(sys.argv[1], "r").read()
out = ast.parse(stuff)

fun = None
for f in out.body:
    if not isinstance(f, ast.FunctionDef):
        continue
    if f.name == "live_analyze":
        fun = f

if fun is None:
    raise ValueError("failed to find the 'live_analyze()' function in 'analyze.py'")

translations = {
    "clust": "clustering",
    "dimred": "dimensionality_reduction",
    "feat": "feature_selection",
    "mark": "marker_detection",
    "nn": "nearest_neighbors",
    "norm": "normalization",
    "qc": "quality_control"
}

def trawl(expr):
    if isinstance(expr, ast.Expr):
        trawl(expr.value)

    elif isinstance(expr, ast.Call):
        if isinstance(expr.func, ast.Name):
            if expr.func.id == "run_neighbor_suite" or expr.func.id == "update":
                expr.func = ast.Attribute(value = ast.Name("scranpy"), attr = expr.func.id)
        else:
            trawl(expr.func)
        for i in range(len(expr.args)):
            trawl(expr.args[i])

    elif isinstance(expr, ast.Name):
        if expr.id == "rna_ptr":
            expr.id = "rna_matrix"
        elif expr.id == "adt_ptr":
            expr.id = "adt_matrix"
        elif expr.id == "crispr_ptr":
            expr.id = "crispr_matrix"

    elif isinstance(expr, ast.Assign):
        trawl(expr.value)

    elif isinstance(expr, ast.Attribute):
        if isinstance(expr.value, ast.Name):
            if expr.value.id in translations:
                expr.value = ast.Attribute(value = ast.Name("scranpy"), attr = translations[expr.value.id])
        else:
            trawl(expr.value)

    elif isinstance(expr, ast.Dict):
        for i in range(len(expr.keys)):
            trawl(expr.keys[i])
        for i in range(len(expr.values)):
            trawl(expr.values[i])

    elif isinstance(expr, ast.Constant):
        pass

    elif isinstance(expr, ast.If):
        new_body = []
        for i in range(len(expr.body)):
            if not trawl(expr.body[i]):
                new_body.append(capture(expr.body[i]))
            else:
                new_body.append(expr.body[i])
        expr.body = new_body

        else_body = []
        for x in expr.orelse:
            if not trawl(x):
                else_body.append(capture(x))
            else:
                else_body.append(x)
        expr.orelse = else_body
        return True

    elif isinstance(expr, ast.For):
        new_body = []
        for i in range(len(expr.body)):
            if not trawl(expr.body[i]):
                new_body.append(capture(expr.body[i]))
            else:
                new_body.append(expr.body[i])
        expr.body = new_body
        return True

    else:
        pass
        #raise TypeError("don't yet know how to handle an 'ast." + expr.__class__.__name__ +"' instance at:\n" + ast.unparse(expr))

    return False

def capture(expr):
    return ast.Expr(
        ast.Call(
            func = ast.Attribute(
                attr = "append",
                value = ast.Name("__commands")
            ),
            args = [ast.Constant(ast.unparse(expr))],
            keywords = []
        )
    )

capturing = False
new_body = [
    ast.parse("""_do_rna = rna_matrix is not None
_do_adt = adt_matrix is not None
_do_crispr = crispr_matrix is not None""")
]

for expr in fun.body:
    if not capturing:
        if isinstance(expr, ast.Assign):
            if isinstance(expr.targets[0], ast.Name) and expr.targets[0].id == "results":
                capturing = True
                new_body.append(ast.parse("__commands = ['import scranpy', 'import numpy', '']").body[0])
                new_body.append(capture(expr))
    elif isinstance(expr, ast.Assign) and len(expr.targets) == 1 and isinstance(expr.targets[0], ast.Name) and expr.targets[0].id.startswith("_"):
        # Underscore variables are transferred directly to the function.
        new_body.append(expr)
    elif not isinstance(expr, ast.Return):
        if not trawl(expr):
            new_body.append(capture(expr))
        else:
            new_body.append(expr)


new_body.append(
    ast.Return(
        value = ast.Call(
            func = ast.Attribute(
                attr = "join",
                value = ast.Constant('\n')
            ),
            args = [ast.Name("__commands")],
            keywords = []
        )
    )
)
fun.returns = ast.Name("str")
fun.body = new_body
fun.name = "dry_analyze"

print("# DO NOT MODIFY THIS FILE! This is automatically generated by '" + sys.argv[0] + "'\n# from the source file '" + sys.argv[1] + "', modify that instead and rerun the script.\n")
print("from .AnalyzeOptions import AnalyzeOptions\n")
print(ast.unparse(fun))
