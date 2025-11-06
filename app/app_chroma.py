"""
FastAPI para RAG usando **Chroma** como almacén vectorial local (sin login)

Endpoints:
- POST /build   → Lee chunks.jsonl, genera embeddings (SBERT) y upsert en Chroma (persist_directory)
- POST /search  → Consulta top‑k en la colección de Chroma

Requisitos:
    pip install fastapi uvicorn pydantic chromadb sentence-transformers

Ejecutar:
    uvicorn app_chroma:app --reload --port 8000

Ejemplos:
    curl -X POST http://localhost:8000/build \
      -H "Content-Type: application/json" \
      -d '{"chunks_path":"./out/chunks.jsonl","outdir":"./chroma_store","collection":"rag_chunks","model":"paraphrase-multilingual-MiniLM-L12-v2","metric":"cosine"}'

    curl -X POST http://localhost:8000/search \
      -H "Content-Type: application/json" \
      -d '{"outdir":"./chroma_store","collection":"rag_chunks","query":"¿Cuál es el objetivo del documento?","topk":5,"model":"paraphrase-multilingual-MiniLM-L12-v2","metric":"cosine"}'
"""

from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Any
import os
import json

from embeddings_chroma import ChromaIndex, load_chunks_jsonl

app = FastAPI(title="RAG API (Chroma)", version="1.0")

# --------- Schemas --------- #


class BuildRequest(BaseModel):
    chunks_path: str = Field(..., description="Ruta a chunks.jsonl")
    outdir: str = Field(..., description="Carpeta de almacenamiento de Chroma")
    collection: str = Field("rag_chunks", description="Nombre de la colección")
    model: str = Field(
        "paraphrase-multilingual-MiniLM-L12-v2", description="Modelo SBERT"
    )
    metric: str = Field("cosine", description="Métrica (cosine|l2|ip)")


class BuildResponse(BaseModel):
    outdir: str
    collection: str
    model: str
    metric: str
    count: int


class SearchRequest(BaseModel):
    outdir: str = Field(..., description="Carpeta de almacenamiento de Chroma")
    collection: str = Field("rag_chunks", description="Nombre de la colección")
    query: str = Field(..., description="Consulta de usuario")
    topk: int = Field(5, ge=1, le=100)
    model: str = Field(
        "paraphrase-multilingual-MiniLM-L12-v2",
        description="Modelo SBERT (debe coincidir)",
    )
    metric: str = Field(
        "cosine", description="Métrica (debe coincidir con la colección)"
    )


class Hit(BaseModel):
    score: float
    id: str
    text: str


class SearchResponse(BaseModel):
    results: List[Hit]


# --------- Endpoints --------- #


@app.post("/build", response_model=BuildResponse)
def build_index(req: BuildRequest):
    if not os.path.exists(req.chunks_path):
        raise HTTPException(status_code=400, detail="chunks_path no existe")
    chunks = load_chunks_jsonl(req.chunks_path)
    if not chunks:
        raise HTTPException(status_code=400, detail="chunks.jsonl vacío o inválido")

    idx = ChromaIndex(
        outdir=req.outdir,
        collection=req.collection,
        metric=req.metric,
        model_name=req.model,
    )
    idx.build_from_chunks(chunks)

    return BuildResponse(
        outdir=os.path.abspath(req.outdir),
        collection=req.collection,
        model=req.model,
        metric=req.metric,
        count=len(chunks),
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    try:
        idx = ChromaIndex(
            outdir=req.outdir,
            collection=req.collection,
            metric=req.metric,
            model_name=req.model,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"No se pudo abrir la colección: {e}"
        )

    res = idx.query(req.query, req.topk)
    hits = [Hit(score=float(s), id=str(i), text=t) for (s, i, t) in res]
    return SearchResponse(results=hits)


@app.get("/health")
def health():
    return {"status": "ok"}
