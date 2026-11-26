"""Synchronize notebook code cells with the exported lab8_tasks.py source."""

from __future__ import annotations

import ast
import json
from pathlib import Path


LAB_DIR = Path(__file__).resolve().parent.parent
PY_PATH = LAB_DIR / "lab8_tasks.py"
NOTEBOOK_PATH = LAB_DIR / "lab8_tasks.ipynb"


def _is_markdown_node(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def extract_code_cells(source: str) -> list[str]:
    """Extract top-level code chunks between exported markdown cells."""
    tree = ast.parse(source)
    source_lines = source.splitlines()
    chunks: list[str] = []
    current: list[ast.AST] = []

    for node in tree.body:
        if _is_markdown_node(node):
            if current:
                chunks.append(_render_chunk(source_lines, current))
                current = []
            continue
        current.append(node)

    if current:
        chunks.append(_render_chunk(source_lines, current))
    return chunks


def _render_chunk(source_lines: list[str], nodes: list[ast.AST]) -> str:
    first_line = nodes[0].lineno - 1
    last_line = nodes[-1].end_lineno
    lines = source_lines[first_line:last_line]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + "\n"


def main() -> None:
    code_chunks = extract_code_cells(PY_PATH.read_text())
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    code_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    if len(code_chunks) != len(code_cells):
        raise RuntimeError(
            f"Expected {len(code_cells)} code chunks, found {len(code_chunks)}"
        )

    for cell, chunk in zip(code_cells, code_chunks, strict=True):
        compile(chunk, f"{NOTEBOOK_PATH}:{cell.get('id', '?')}", "exec")
        if isinstance(cell.get("source"), str):
            cell["source"] = chunk
        else:
            cell["source"] = chunk.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None

    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
    )
    print(f"Synchronized {len(code_cells)} notebook code cells")


if __name__ == "__main__":
    main()
