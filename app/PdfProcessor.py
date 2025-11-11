"""
PDF → Texto → Chunks (para RAG) **solo PDFs con texto** (sin OCR)

Componentes:
- PdfProcessor.extract_text(): extrae texto de un PDF con PyMuPDF.
- split_text_recursive(): divide en trozos por separadores con solape.
- split_by_tokens(): divide por tokens (si hay tiktoken). Fallback a chars.

CLI rápido:
    python pdf_reader_chunker.py archivo.pdf --out chunks.jsonl --size 1200 --overlap 200
    python pdf_reader_chunker.py archivo.pdf --tokens --size 400 --overlap 60 --out chunks_tok.jsonl

Dependencias:
    pip install pymupdf tiktoken
"""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

# ===============================
# Procesador PDF (texto nativo)
# ===============================


@dataclass
class PdfProcessor:
    def extract_text(self, path: str) -> str:
        """Extrae texto de un PDF usando PyMuPDF (fitz).
        Devuelve el texto concatenado con separadores por página.
        """
        try:
            import fitz  # PyMuPDF
        except Exception as e:
            raise RuntimeError(
                "PyMuPDF no está instalado. Ejecuta 'pip install pymupdf'. Error: %r"
                % e
            )
        if not os.path.exists(path):
            raise FileNotFoundError(f"No existe el archivo: {path}")

        doc = fitz.open(path)
        parts: List[str] = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            text = text.strip()
            parts.append(f"\n\n=== Page {i} ===\n\n{text}")
        doc.close()
        return "".join(parts).strip()


# -------- Particionado en chunks (caracteres) -------- #


def _split_once(text: str, sep: str) -> List[str]:
    return [t for t in text.split(sep) if t]


def _merge_with_limit(pieces: Iterable[str], limit: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush():
        nonlocal current, current_len
        if not current:
            return
        chunk = "".join(current).strip()
        if chunk:
            chunks.append(chunk)
        if overlap > 0 and chunk:
            keep = max(0, overlap)
            tail = chunk[-keep:]
            current = [tail]
            current_len = len(tail)
        else:
            current = []
            current_len = 0

    for piece in pieces:
        piece_len = len(piece)
        if current_len + piece_len <= limit or not current:
            current.append(piece)
            current_len += piece_len
        else:
            flush()
            current = [piece]
            current_len = piece_len
    flush()
    return chunks


def split_text_recursive(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
    separators: Optional[List[str]] = None,
) -> List[str]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    def _recursive(parts: List[str], seps: List[str]) -> List[str]:
        if not seps:
            base = []
            for p in parts:
                for i in range(0, len(p), chunk_size):
                    base.append(p[i : i + chunk_size])
            return base
        sep = seps[0]
        next_seps = seps[1:]
        new_parts: List[str] = []
        for p in parts:
            if len(p) <= chunk_size:
                new_parts.append(p)
                continue
            pieces = _split_once(p, sep)
            if len(pieces) == 1:
                deeper = _recursive([p], next_seps)
                new_parts.extend(deeper)
            else:
                deeper = _recursive(pieces, next_seps)
                new_parts.extend(deeper)
        return new_parts

    fragmented = _recursive([text], separators)
    chunks = _merge_with_limit(fragmented, chunk_size, overlap)
    return [c for c in chunks if c.strip()]


# -------- Particionado por tokens (opcional con tiktoken) -------- #


def split_by_tokens(
    text: str,
    chunk_tokens: int = 400,
    overlap: int = 60,
    encoding_name: str = "cl100k_base",
) -> List[str]:
    try:
        import tiktoken
    except Exception:
        return split_text_recursive(
            text, chunk_size=chunk_tokens * 4, overlap=overlap * 4
        )

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    step = max(1, chunk_tokens - overlap)
    chunks: List[str] = []
    for start in range(0, len(tokens), step):
        end = min(start + chunk_tokens, len(tokens))
        piece = enc.decode(tokens[start:end])
        if piece.strip():
            chunks.append(piece)
    return chunks


# -------- Utilidad de guardado -------- #


def _save_jsonl(chunks: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            rec = {"id": i, "text": c}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -------- CLI -------- #


def main():
    parser = argparse.ArgumentParser(description="PDF → Texto → Chunks (sin OCR)")
    parser.add_argument("pdf", type=str, help="Ruta del PDF")
    parser.add_argument(
        "--out", type=str, default="chunks.jsonl", help="Ruta de salida JSONL"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1200,
        help="Tamaño máximo de chunk (caracteres o ≈tokens*4)",
    )
    parser.add_argument("--overlap", type=int, default=200, help="Solape entre chunks")
    parser.add_argument(
        "--tokens", action="store_true", help="Dividir por tokens (requiere tiktoken)"
    )

    args = parser.parse_args()
    proc = PdfProcessor()

    text = proc.extract_text(args.pdf)

    if args.tokens:
        chunks = split_by_tokens(text, chunk_tokens=args.size, overlap=args.overlap)
    else:
        chunks = split_text_recursive(text, chunk_size=args.size, overlap=args.overlap)

    _save_jsonl(chunks, args.out)

    stats = {
        "pdf": args.pdf,
        "num_chars": len(text),
        "num_chunks": len(chunks),
        "avg_chunk_len": int(sum(len(c) for c in chunks) / max(1, len(chunks))),
        "out": os.path.abspath(args.out),
        "mode": "tokens" if args.tokens else "chars",
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
