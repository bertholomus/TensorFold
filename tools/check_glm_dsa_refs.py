"""Static check: every `alias.attr` used in the glm_moe_dsa CUDA family resolves on the imported module.

Imports need torch/triton, so this runs inside the pytorch container (CPU is enough for attribute lookup).
"""
import ast, importlib, pathlib, sys

PKG = pathlib.Path("/tfw/TensorFold/src/tensorfold/families/glm_moe_dsa/cuda")
bad = 0
for f in sorted(PKG.glob("*.py")):
    tree = ast.parse(f.read_text())
    modname = "tensorfold.families.glm_moe_dsa.cuda." + f.stem
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = "tensorfold.families.glm_moe_dsa.cuda" + ("." + base if base else "")
            for a in node.names:
                full = f"{base}.{a.name}"
                try:
                    importlib.import_module(full)
                    aliases[a.asname or a.name] = full
                except Exception:
                    pass          # a symbol import, not a module
        elif isinstance(node, ast.Import):
            for a in node.names:
                aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            mod = aliases[node.value.id]
            if mod.split(".")[0] in ("torch", "triton", "tl", "np", "numpy", "os", "math", "sys", "json", "re",
                                     "threading", "time", "struct", "ctypes"):
                continue
            try:
                m = importlib.import_module(mod)
            except Exception as e:
                print(f"{f.name}:{node.lineno} import {mod} failed: {e}")
                bad += 1
                continue
            if not hasattr(m, node.attr):
                print(f"{f.name}:{node.lineno} {node.value.id}.{node.attr} missing on {mod}")
                bad += 1
    # names imported via `from X import a, b`
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = "tensorfold.families.glm_moe_dsa.cuda" + ("." + base if base else "")
            try:
                m = importlib.import_module(base)
            except Exception as e:
                print(f"{f.name}:{node.lineno} from {base}: {e}")
                bad += 1
                continue
            for a in node.names:
                if a.name != "*" and not hasattr(m, a.name):
                    try:
                        importlib.import_module(f"{base}.{a.name}")
                    except Exception:
                        print(f"{f.name}:{node.lineno} from {base} import {a.name}: missing")
                        bad += 1
print("BAD", bad)
