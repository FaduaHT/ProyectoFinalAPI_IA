"""
Embeddings + Chroma (local, sin login)

• Lee `chunks` (id, text), genera embeddings con Sentence-Transformers y
  los persiste en una base local de Chroma.
• Permite consultas top‑k (cosine) desde CLI.

Instalación:
    pip install chromadb sentence-transformers

Uso:
    # 1) construir / actualizar la colección
    python embeddings_chroma.py build \
      --chunks ./out/chunks.jsonl \
      --outdir ./chroma_store \
      --collection rag_chunks \
      --model paraphrase-multilingual-MiniLM-L12-v2

    # 2) consultar
    python embeddings_chroma.py query \
      --outdir ./chroma_store \
      --collection rag_chunks \
      --q "¿Cuál es el objetivo del documento?" \
      --topk 5

Notas:
- Usa Chroma **PersistentClient** → no requiere cuentas ni login; guarda en `--outdir`.
- Métrica: cosine (HNSW). Puedes cambiar con --metric (cosine|l2|ip).
"""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

# deps: chroma + sentence-transformers
try:
    import chromadb
    from chromadb.config import Settings
except Exception as e:
    raise RuntimeError("Instala chromadb: pip install chromadb. Error: %r" % e)

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError(
        "Instala sentence-transformers: pip install sentence-transformers. Error: %r"
        % e
    )


# ----------------- utilidades E/S ----------------- #


def load_chunks_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe: {path}")
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("id")
            txt = rec.get("text")
            if txt is None:
                continue
            out.append({"id": cid, "text": txt})
    return out


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ----------------- motor embeddings ----------------- #


@dataclass
class EmbeddingModel:
    name: str = "paraphrase-multilingual-MiniLM-L12-v2"

    def __post_init__(self):
        self.model = SentenceTransformer(self.name)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,  # dejamos la normalización a Chroma/metric
        )
        return vecs.astype("float32")


# ----------------- índice Chroma ----------------- #


@dataclass
class ChromaIndex:
    outdir: str
    collection: str = "rag_chunks"
    metric: str = "cosine"  # "cosine" | "l2" | "ip"
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"

    def __post_init__(self):
        os.makedirs(self.outdir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.outdir, settings=Settings())
        self.collection = self.client.get_or_create_collection(
            name=self.collection,
            metadata={"hnsw:space": self.metric},
        )
        self.embedder = EmbeddingModel(self.model_name)

    # ---- build / upsert ---- #
    def build_from_chunks(self, chunks: List[Dict[str, Any]]):
        texts = [c["text"] for c in chunks]
        ids = [
            str(c["id"]) if c.get("id") is not None else str(i)
            for i, c in enumerate(chunks)
        ]
        embeddings = self.embedder.encode(texts).tolist()

        # Upsert (evita duplicados por id)
        self.collection.upsert(ids=ids, documents=texts, embeddings=embeddings)

        # Guardar metadatos mínimos
        save_json(
            os.path.join(self.outdir, "chroma_meta.json"),
            {
                "collection": self.collection.name,
                "metric": self.metric,
                "model": self.model_name,
                "count": len(ids),
            },
        )

    # ---- query ---- #
    def query(self, q: str, topk: int = 5) -> List[Tuple[float, str, str]]:
        qv = self.embedder.encode([q]).tolist()
        res = self.collection.query(
            query_embeddings=qv,
            n_results=topk,
            include=["documents", "distances"],  # ← SIN "ids"
        )
        # Chroma devuelve ids aunque no estén en include
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0]
        ids = res.get("ids", [[]])[0]

        results: List[Tuple[float, str, str]] = []
        for doc_id, doc, dist in zip(ids, docs, dists):
            score = 1.0 - float(dist) if self.metric == "cosine" else float(-dist)
            results.append((score, str(doc_id), doc))
        return results


# ----------------- CLI ----------------- #


def _cmd_build(args: argparse.Namespace):
    chunks = load_chunks_jsonl(args.chunks)
    idx = ChromaIndex(
        outdir=args.outdir,
        collection=args.collection,
        metric=args.metric,
        model_name=args.model,
    )
    idx.build_from_chunks(chunks)
    print(
        json.dumps(
            {
                "outdir": os.path.abspath(args.outdir),
                "collection": args.collection,
                "model": args.model,
                "metric": args.metric,
                "count": len(chunks),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _cmd_query(args: argparse.Namespace):
    idx = ChromaIndex(
        outdir=args.outdir,
        collection=args.collection,
        metric=args.metric,
        model_name=args.model,
    )
    res = idx.query(args.q, args.topk)
    payload = [
        {
            "score": round(float(s), 4),
            "id": cid,
            "text": text[:300] + ("…" if len(text) > 300 else ""),
        }
        for (s, cid, text) in res
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Embeddings + Chroma (local)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser(
        "build", help="Construir/actualizar colección desde chunks.jsonl"
    )
    p_build.add_argument(
        "--chunks", required=True, type=str, help="Ruta a chunks.jsonl"
    )
    p_build.add_argument(
        "--outdir", required=True, type=str, help="Carpeta de almacenamiento de Chroma"
    )
    p_build.add_argument(
        "--collection", default="rag_chunks", type=str, help="Nombre de la colección"
    )
    p_build.add_argument(
        "--model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        type=str,
        help="Modelo SBERT",
    )
    p_build.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "l2", "ip"],
        help="Métrica de similitud",
    )
    p_build.set_defaults(func=_cmd_build)

    p_query = sub.add_parser("query", help="Consultar colección")
    p_query.add_argument(
        "--outdir", required=True, type=str, help="Carpeta de almacenamiento de Chroma"
    )
    p_query.add_argument(
        "--collection", default="rag_chunks", type=str, help="Nombre de la colección"
    )
    p_query.add_argument("--q", required=True, type=str, help="Pregunta / consulta")
    p_query.add_argument("--topk", default=5, type=int, help="Número de resultados")
    p_query.add_argument(
        "--model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        type=str,
        help="Modelo SBERT",
    )
    p_query.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "l2", "ip"],
        help="Métrica de similitud (debe coincidir)",
    )
    p_query.set_defaults(func=_cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
