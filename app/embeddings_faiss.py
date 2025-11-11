"""
Embeddings + FAISS (local, sin API keys)

• Lee `chunks.jsonl` (id, text) y genera embeddings con un modelo multilingual (ES compatible).
• Crea un índice FAISS (cosine) y guarda: `faiss.index` + `faiss_meta.json`.
• Permite consultas top‑k desde CLI.

Requisitos:
    pip install sentence-transformers faiss-cpu
(En macOS con Apple Silicon puede ser `pip install faiss-cpu` o `faiss` según entorno.)

Uso:
    # 1) construir el índice a partir de chunks.jsonl
    python embeddings_faiss.py build --chunks ./out/chunks.jsonl --outdir ./faiss --model paraphrase-multilingual-MiniLM-L12-v2

    # 2) consultar
    python embeddings_faiss.py query --outdir ./faiss --q "¿Cuál es el objetivo del documento?" --topk 5

Notas:
- Modelo por defecto: paraphrase-multilingual-MiniLM-L12-v2 (bueno para ES y ligero).
- Métrica: coseno usando IndexFlatIP + normalización L2.
"""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np

# FAISS
try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("No se pudo importar FAISS. Instala 'faiss-cpu'. Error: %r" % e)

# Sentence-Transformers
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError(
        "No se pudo importar sentence-transformers. Instala 'sentence-transformers'. Error: %r"
        % e
    )


# ---------- Utilidades de E/S ---------- #


def load_chunks_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe: {path}")
    chunks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # tolerante con claves (id/text)
            cid = rec.get("id")
            txt = rec.get("text")
            if txt is None:
                continue
            chunks.append({"id": cid, "text": txt})
    return chunks


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Índice de embeddings ---------- #


@dataclass
class EmbeddingsIndex:
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    normalize: bool = True  # para cosine con IndexFlatIP

    def __post_init__(self):
        self.model = SentenceTransformer(self.model_name)
        self.index = None  # type: ignore
        self.meta: Dict[str, Any] = {}

    # ---- embedding helpers ---- #
    def _embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        vecs = vecs.astype("float32")
        if self.normalize:
            faiss.normalize_L2(vecs)
        return vecs

    # ---- build ---- #
    def build_from_chunks(self, chunks: List[Dict[str, Any]]):
        texts = [c["text"] for c in chunks]
        ids = [c.get("id", i) for i, c in enumerate(chunks)]
        vecs = self._embed(texts)

        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vecs)

        # guardar metadata mínima (id → {text})
        self.meta = {
            "ids": ids,
            "count": len(ids),
            "model": self.model_name,
            "normalize": self.normalize,
        }

    # ---- persistencia ---- #
    def save(self, outdir: str):
        if self.index is None:
            raise RuntimeError("Índice no construido")
        os.makedirs(outdir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(outdir, "faiss.index"))
        save_json(os.path.join(outdir, "faiss_meta.json"), self.meta)

    def load(self, outdir: str):
        idx_path = os.path.join(outdir, "faiss.index")
        meta_path = os.path.join(outdir, "faiss_meta.json")
        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("No se encontró el índice en esa carpeta")
        self.index = faiss.read_index(idx_path)
        self.meta = read_json(meta_path)
        # asegurar que el modelo y normalización coinciden
        self.model = SentenceTransformer(self.meta.get("model", self.model_name))
        self.normalize = bool(self.meta.get("normalize", True))

    # ---- query ---- #
    def query(
        self, q: str, topk: int, chunks: List[Dict[str, Any]]
    ) -> List[Tuple[float, Any, str]]:
        if self.index is None:
            raise RuntimeError("Índice no cargado/construido")
        qv = self._embed([q])
        scores, idxs = self.index.search(qv, topk)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()

        results: List[Tuple[float, Any, str]] = []
        ids = self.meta["ids"]
        for score, pos in zip(scores, idxs):
            if pos < 0:
                continue
            cid = ids[pos]
            text = chunks[pos]["text"]
            results.append((float(score), cid, text))
        return results


# ---------- CLI ---------- #


def _cmd_build(args: argparse.Namespace):
    chunks = load_chunks_jsonl(args.chunks)
    idx = EmbeddingsIndex(model_name=args.model, normalize=not args.no_norm)
    idx.build_from_chunks(chunks)
    idx.save(args.outdir)
    print(
        json.dumps(
            {
                "outdir": os.path.abspath(args.outdir),
                "count": len(chunks),
                "model": args.model,
                "normalized": not args.no_norm,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _cmd_query(args: argparse.Namespace):
    # Cargamos metadatos + índice y también los chunks para devolver texto
    idx = EmbeddingsIndex(model_name=args.model, normalize=not args.no_norm)
    idx.load(args.outdir)
    # Releer chunks desde la misma carpeta si se guardaron allí; si no, usar --chunks
    if args.chunks and os.path.exists(args.chunks):
        chunks = load_chunks_jsonl(args.chunks)
    else:
        # intentar fallback: buscar un chunks.jsonl en outdir
        default_chunks = os.path.join(args.outdir, "chunks.jsonl")
        if os.path.exists(default_chunks):
            chunks = load_chunks_jsonl(default_chunks)
        else:
            raise FileNotFoundError(
                "Debes pasar --chunks con la ruta al JSONL de chunks"
            )

    res = idx.query(args.q, args.topk, chunks)
    # salida bonita
    payload = []
    for score, cid, text in res:
        payload.append(
            {
                "score": round(float(score), 4),
                "id": cid,
                "text": text[:300] + ("…" if len(text) > 300 else ""),
            }
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Embeddings + FAISS (local)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Construir índice desde chunks.jsonl")
    p_build.add_argument(
        "--chunks", required=True, type=str, help="Ruta a chunks.jsonl"
    )
    p_build.add_argument(
        "--outdir", required=True, type=str, help="Carpeta de salida del índice"
    )
    p_build.add_argument(
        "--model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        type=str,
        help="Modelo SBERT",
    )
    p_build.add_argument(
        "--no-norm",
        action="store_true",
        help="Desactivar normalización L2 (no recomendado)",
    )
    p_build.set_defaults(func=_cmd_build)

    p_query = sub.add_parser("query", help="Consultar el índice")
    p_query.add_argument(
        "--outdir", required=True, type=str, help="Carpeta del índice guardado"
    )
    p_query.add_argument(
        "--q", required=True, type=str, help="Pregunta o texto de consulta"
    )
    p_query.add_argument("--topk", default=5, type=int, help="Número de resultados")
    p_query.add_argument(
        "--chunks",
        default=None,
        type=str,
        help="Ruta a chunks.jsonl (para devolver texto)",
    )
    p_query.add_argument(
        "--model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        type=str,
        help="Modelo SBERT",
    )
    p_query.add_argument(
        "--no-norm",
        action="store_true",
        help="Desactivar normalización L2 (no recomendado)",
    )
    p_query.set_defaults(func=_cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
