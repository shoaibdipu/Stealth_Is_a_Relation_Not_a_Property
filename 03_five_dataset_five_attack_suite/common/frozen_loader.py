"""Load class/function definitions from procedural frozen experiment scripts without running them."""
import ast
from pathlib import Path

def load_definitions(script_path, overrides=None, seed_globals=None):
    path = Path(script_path)
    tree = ast.parse(path.read_text(), filename=str(path))
    ns = {"__name__": f"frozen_{path.stem}", "__file__": str(path)}
    if seed_globals:
        ns.update(seed_globals)
    overrides = dict(overrides or {})
    ns.update(overrides)
    deferred = []
    for node in tree.body:
        is_def = isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        simple = bool(targets) and all(isinstance(t, ast.Name) for t in targets)
        names = [t.id for t in targets] if simple else []
        keep_assign = simple and (all(n.isupper() for n in names) or not any(isinstance(x, ast.Call) for x in ast.walk(node.value)))
        if not (is_def or keep_assign):
            continue
        if keep_assign and any(n in overrides for n in names):
            continue
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
        except NameError:
            # Function defaults can depend on constants that occur inside skipped if-blocks.
            # Retry after the first pass once explicit overrides/default seed globals exist.
            deferred.append(node)
        except Exception:
            if is_def:
                raise
    for node in deferred:
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
        except Exception:
            pass
    return ns

def merge_definitions(paths, overrides=None):
    ns = {}
    for p in paths:
        cur = load_definitions(p, overrides=overrides, seed_globals=ns)
        ns.update(cur)
    return ns
