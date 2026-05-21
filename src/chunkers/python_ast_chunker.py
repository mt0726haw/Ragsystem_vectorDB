"""Python AST-based chunker.

Produces one chunk per top-level function/class and per method inside a class.
A residual chunk with module-level statements (imports, constants, docstring)
is emitted as well so module-wide context isn't lost.
"""
from __future__ import annotations

import ast
import logging
from typing import List, Optional

from src.model import Chunk, Document

logger = logging.getLogger(__name__)


class PythonAstChunker:
    """Chunks Python code based on its AST structure."""

    def chunk(self, document: Document) -> List[Chunk]:
        source = document.content
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            logger.warning(
                "SyntaxError in %s (%s) - falling back to single chunk", document.file_name, exc
            )
            return self._fallback(document)

        source_lines = source.splitlines()
        module_name = document.file_path.stem
        chunks: List[Chunk] = []

        # Module docstring + module-level non-def statements
        module_chunk = self._module_chunk(tree, source_lines, document, module_name)
        if module_chunk:
            chunks.append(module_chunk)

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(
                    self._chunk_from_node(node, source_lines, document, module_name, class_name=None)
                )
            elif isinstance(node, ast.ClassDef):
                # Class-level chunk (signature + docstring + bases)
                chunks.append(
                    self._chunk_from_node(node, source_lines, document, module_name, class_name=None)
                )
                # One chunk per method
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        chunks.append(
                            self._chunk_from_node(
                                sub, source_lines, document, module_name, class_name=node.name
                            )
                        )

        logger.debug("Python AST chunker produced %d chunks for %s", len(chunks), document.file_name)
        return [c for c in chunks if c.text.strip()]

    def _chunk_from_node(
        self,
        node: ast.AST,
        source_lines: List[str],
        document: Document,
        module_name: str,
        class_name: Optional[str],
    ) -> Chunk:
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start)
        body = "\n".join(source_lines[start - 1 : end])

        kind = type(node).__name__
        name = getattr(node, "name", "<anonymous>")

        metadata = {
            "module": module_name,
            "class": class_name,
            "function": name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None,
            "node_type": kind,
            "line_start": start,
            "line_end": end,
            "source_file": document.file_name,
        }
        if isinstance(node, ast.ClassDef):
            metadata["function"] = None
            metadata["class"] = node.name

        return Chunk(
            text=body,
            source_file=str(document.file_path),
            category=document.category,
            metadata=metadata,
        )

    def _module_chunk(
        self,
        tree: ast.Module,
        source_lines: List[str],
        document: Document,
        module_name: str,
    ) -> Optional[Chunk]:
        module_only_nodes = [
            n for n in tree.body
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if not module_only_nodes:
            # Still emit module docstring if available
            docstring = ast.get_docstring(tree)
            if not docstring:
                return None
            return Chunk(
                text=f'"""{docstring}"""',
                source_file=str(document.file_path),
                category=document.category,
                metadata={
                    "module": module_name,
                    "class": None,
                    "function": None,
                    "node_type": "ModuleDocstring",
                    "line_start": 1,
                    "line_end": 1,
                    "source_file": document.file_name,
                },
            )

        start = min(getattr(n, "lineno", 1) for n in module_only_nodes)
        end = max(getattr(n, "end_lineno", start) for n in module_only_nodes)
        body = "\n".join(source_lines[start - 1 : end])
        return Chunk(
            text=body,
            source_file=str(document.file_path),
            category=document.category,
            metadata={
                "module": module_name,
                "class": None,
                "function": None,
                "node_type": "ModuleLevel",
                "line_start": start,
                "line_end": end,
                "source_file": document.file_name,
            },
        )

    @staticmethod
    def _fallback(document: Document) -> List[Chunk]:
        return [
            Chunk(
                text=document.content,
                source_file=str(document.file_path),
                category=document.category,
                metadata={
                    "module": document.file_path.stem,
                    "class": None,
                    "function": None,
                    "node_type": "ParseError",
                    "source_file": document.file_name,
                },
            )
        ]
