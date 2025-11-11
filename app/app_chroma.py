"""
FastAPI para RAG usando **Chroma** como almacén vectorial local (sin login)

Endpoints:
- POST /build   → Lee chunks.jsonl, genera embeddings (SBERT) y upsert en Chroma (persist_directory)
- POST /search  → Consulta top-k en la colección de Chroma
- POST /answer  → Recupera contexto y llama a Groq para generar respuesta con fuentes

Requisitos:
    pip install fastapi uvicorn pydantic chromadb sentence-transformers groq python-dotenv

Ejecutar:
    uvicorn app_chroma:app --reload --port 8000

Ejemplos:
    curl -X POST http://localhost:8000/build \
      -H "Content-Type: application/json" \
      -d '{"chunks_path":"./out/chunks.jsonl","outdir":"./chroma_store","collection":"rag_chunks","model":"paraphrase-multilingual-MiniLM-L12-v2","metric":"cosine"}'

    curl -X POST http://localhost:8000/search \
      -H "Content-Type: application/json" \
      -d '{"outdir":"./chroma_store","collection":"rag_chunks","query":"¿Cuál es el objetivo del documento?","topk":5,"model":"paraphrase-multilingual-MiniLM-L12-v2","metric":"cosine"}'

    curl -X POST http://localhost:8000/answer \
      -H "Content-Type: application/json" \
      -d '{"outdir":"./chroma_store","collection":"rag_chunks","query":"Fases de vida del plan municipal","top_k":5,"model":"llama-3.1-70b-versatile","temperature":0.1,"max_tokens":700}'
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Cargar variables de entorno desde .env (GROQ_API_KEY, etc.)
load_dotenv()

# Servicios propios (LLM y utilidades RAG)
from services.llm_groq import chat_complete
from services.rag_utils import format_context, build_messages

# Wrapper local para construir índice (ingesta)
from embeddings_chroma import ChromaIndex, load_chunks_jsonl

app = FastAPI(title="RAG API (Chroma)", version="1.1")

# -------------------------------------------------------------------
# Utilidades Chroma (cliente y recuperación unificada)
# -------------------------------------------------------------------


def _get_chroma_collection(outdir: str, collection_name: str):
    """
    Devuelve la colección de Chroma, siendo compatible con chromadb >=0.5 (PersistentClient)
    y anteriores (Client + Settings). No crea duplicados si ya existe.
    """
    try:
        # chromadb >= 0.5
        from chromadb import PersistentClient

        client = PersistentClient(path=outdir)
        col = client.get_or_create_collection(name=collection_name)
        return col
    except Exception:
        # fallback a versiones anteriores
        from chromadb import Client
        from chromadb.config import Settings

        client = Client(Settings(persist_directory=outdir))
        col = client.get_or_create_collection(name=collection_name)
        return col


def retrieve_context_full(outdir: str, collection_name: str, query: str, top_k: int):
    """
    Recupera de Chroma: documentos, metadatas, distances e ids (si los expone la versión).
    Compatible con versiones que NO permiten 'ids' en include.
    """
    col = _get_chroma_collection(outdir=outdir, collection_name=collection_name)

    # NO pedir 'ids' en include (algunas versiones lo rechazan)
    res = col.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],  # <- sin 'ids'
    )

    docs = (res.get("documents") or [[]])[0]
    raw_metas = (res.get("metadatas") or [[]])[0]
    metas = [(m or {}) for m in raw_metas]  # <--- clave
    dists = (res.get("distances") or [[]])[0]
    ids = (
        (res.get("ids") or [[]])[0]
        if res.get("ids")
        else [f"doc_{i+1}" for i in range(len(docs))]
    )
    return docs, metas, dists, ids


# -------------------------------------------------------------------
# Schemas
# -------------------------------------------------------------------


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
    topk: int = Field(5, ge=1, le=100)  # mantener compatibilidad con tu /search
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


class AnswerRequest(BaseModel):
    # Recuperación (mismos campos que /search para unificar)
    outdir: str = Field(
        "./chroma_store", description="Carpeta de almacenamiento de Chroma"
    )
    collection: str = Field("rag_chunks", description="Nombre de la colección")
    query: str = Field(..., description="Pregunta del usuario")
    top_k: int = Field(5, ge=1, le=50)

    # LLM (Groq)
    model: str = Field("llama-3.1-70b-versatile", description="Modelo LLM Groq")
    temperature: float = Field(0.2, ge=0.0, le=1.0)
    max_tokens: int = Field(512, ge=64, le=4096)


class SourceItem(BaseModel):
    idx: int
    source: Optional[str] = None
    page: Optional[str] = None
    score: float
    snippet: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    used_model: str
    sources: List[SourceItem]


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------


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
    """
    Devuelve hits "en crudo" (score/id/text) para depuración o UI avanzada.
    Internamente usa la misma recuperación que /answer.
    """
    try:
        docs, metas, dists, ids = retrieve_context_full(
            outdir=req.outdir,
            collection_name=req.collection,
            query=req.query,
            top_k=req.topk,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"No se pudo abrir/consultar la colección: {e}"
        )

    # Construimos hits a partir de la recuperación nativa:
    hits = [
        Hit(score=float(s), id=str(i), text=(t or ""))
        for (s, i, t) in zip(dists, ids, docs)
    ]
    return SearchResponse(results=hits)


@app.post("/answer", response_model=AnswerResponse, tags=["RAG Answer"])
def get_answer(body: AnswerRequest):
    """
    Un solo paso para el cliente: recupera contexto y genera respuesta con Groq.
    """
    q = (body.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="La query no puede estar vacía.")

    # 1) Recuperación desde Chroma (compartida con /search)
    try:
        docs, metas, dists, _ids = retrieve_context_full(
            outdir=body.outdir,
            collection_name=body.collection,
            query=q,
            top_k=body.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error consultando Chroma: {e}")

    if not docs:
        return AnswerResponse(
            question=q,
            answer="No he encontrado información relevante en tus documentos para responder a esta pregunta.",
            used_model=body.model,
            sources=[],
        )

    # 2) Preparar contexto legible + mensajes para el LLM
    context_str = format_context(docs, metas, dists)
    messages = build_messages(q, context_str)

    # 3) Llamada a Groq (usa tu helper services.llm_groq.chat_complete)
    try:
        answer = chat_complete(
            messages=messages,
            model=body.model,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error llamando a Groq: {e}")

    # 4) Fuentes compactas
    sources: List[SourceItem] = []
    for i, (txt, meta, score) in enumerate(zip(docs, metas, dists), start=1):
        safe_meta = meta or {}
        sources.append(
            SourceItem(
                idx=i,
                source=(
                    safe_meta.get("source")
                    or safe_meta.get("filename")
                    or safe_meta.get("doc_id")
                ),
                page=str(safe_meta.get("page") or safe_meta.get("page_num") or ""),
                score=float(score) if score is not None else 0.0,
                snippet=(txt or "")[:350],
            )
        )

    return AnswerResponse(
        question=q,
        answer=answer,
        used_model=body.model,
        sources=sources,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
