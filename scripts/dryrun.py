# This script generates a dry-run version of the __analyze function in 
# src/scranpy/analyze.py. The idea is to allow users to create a dry-run
# of all the individual commands that they can copy-paste into their
# scripts, if they want to customize the analysis but they don't want
# to actually deal with learning all of the individual steps.
#
# The meta-parsing rules are as follows:
# - We only start converting code after 'results'.
# - Any non-If, non-For expression is converted into a stringified expression.
# - Any If or For is retained.

import ast 
import astor
stuff = open("analyze.py", "r").read()
out = ast.parse(stuff)

fun = None
for f in out.body:
    if not isinstance(f, ast.FunctionDef):
        continue
    if f.name == "__analyze":
        fun = f

if fun is None:
    raise ValueError("failed to find the '__analyze()' function in 'analyze.py'")

translations = {
    "clust": "clustering",
    "dimred": "dimensionality_reduction",
    "feat": "feature_selection",
    "mark": "marker_detection",
    "nn": "nearest_neighbors",
    "norm": "normalization",
    "qc": "quality_control"
}

def translate(name):
    i = name.find(".")
    if i > 0:
        header = name[:i]
        if header in translations:
            return "scranpy." + translations[header] + name[i:]
    return name

def sanitize(expr):
    if isinstance(expr, ast.Expr):
        sanitize(expr.value)

    elif isinstance(expr, ast.Call):
        sanitize(expr.func)
        for i in range(len(expr.args)):
            sanitize(expr.args[i])

    elif isinstance(expr, ast.Name):
        translate(expr.id)

    elif isinstance(expr, ast.Assign):
        sanitize(expr.value)

    elif isinstance(expr, ast.Attribute):
        sanitize(expr.value)

    elif isinstance(expr, ast.Dict):
        for i in range(len(expr.keys)):
            sanitize(expr.keys[i])
        for i in range(len(expr.values)):
            sanitize(expr.values[i])

    elif isinstance(expr, ast.Constant):
        pass

    else:
        return
        raise TypeError("don't yet know how to handle an 'ast." + expr.__class__.__name__ +"' instance at:\n" + astor.to_source(expr))

def is_capturable(expr):
    if isinstance(expr, ast.If) or isinstance(expr, ast.For) or isinstance(expr, ast.Raise):
        return False
    return True

def capture(expr):
    return ast.Expr(
        ast.Call(
            func = ast.Attribute(
                attr = "append",
                value = ast.Name("__commands")
            ),
            args = [ast.Constant(astor.to_source(expr))],
            keywords = []
        )
    )

def trawl(expr):
    if isinstance(expr, ast.If):
        new_body = []
        for i in range(len(expr.body)):
            trawl(expr.body[i])
            if is_capturable(expr.body[i]):
                new_body.append(capture(expr.body[i]))
            else:
                new_body.append(expr.body[i])
        expr.body = new_body

        for i in range(len(expr.orelse)):
            trawl(expr.orelse[i])

    elif isinstance(expr, ast.For): 
        new_body = []
        for i in range(len(expr.body)):
            trawl(expr.body[i])
            if is_capturable(expr.body[i]):
                new_body.append(capture(expr.body[i]))
            else:
                new_body.append(expr.body[i])
        expr.body = new_body

capturing = False
new_body = []
for expr in fun.body:
    sanitize(expr)
    if not capturing:
        if isinstance(expr, ast.Assign):
            if isinstance(expr.targets[0], ast.Name) and expr.targets[0].id == "results":
                capturing = True
                new_body.append(ast.parse("__commands = []").body[0])
                new_body.append(capture(expr))
    else:
        if not is_capturable(expr):
            trawl(expr)
            new_body.append(expr)
        else:
            new_body.append(capture(expr))

fun.body = new_body

# Peel out the arguments.
new_args = []
new_arg_defaults = []
nargs = len(fun.args.args)
nodef = nargs - len(fun.args.defaults)
for i in range(nargs):
    arg = fun.args.args[i]
    if arg.arg == "options":
        new_args.append(arg)
        new_arg_defaults.append(fun.args.defaults[i - nodef])

fun.args.args = new_args
fun.args.defaults = new_arg_defaults
