"""Produce a comment-free / docstring-free copy of src/ under dist/.

Uses the AST: ast.unparse drops comments (they are not part of the tree) and we
additionally strip module/class/function docstrings.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DIST = ROOT / "dist"


def strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return tree


def main() -> None:
    DIST.mkdir(exist_ok=True)
    count = 0
    for py in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text())
            strip_docstrings(tree)
            code = ast.unparse(tree)
        except Exception:
            code = py.read_text()  # leave un-parseable files as-is
        out = DIST / py.relative_to(ROOT)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(code + "\n")
        count += 1
    print(f"Clean version -> dist/ ({count} files)")


if __name__ == "__main__":
    main()
