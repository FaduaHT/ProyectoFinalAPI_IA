from typing import List, Dict, Any


def format_context(
    docs: List[str], metas: List[Dict[str, Any]], scores: List[float]
) -> str:
    lines = []
    # Asegurar longitudes iguales y tolerancia a None
    n = min(len(docs), len(metas), len(scores))
    for i in range(n):
        txt = docs[i] or ""
        meta = metas[i] or {}  # <--- clave: si viene None, usar dict vacío
        score = scores[i] if i < len(scores) else 0.0

        src = (
            meta.get("source")
            or meta.get("filename")
            or meta.get("doc_id")
            or "desconocido"
        )
        page = meta.get("page") or meta.get("page_num") or "-"

        lines.append(
            f"[{i+1}] Source: {src} | Page: {page} | Score: {float(score):.4f}\n{txt}"
        )
    return "\n\n---\n\n".join(lines)


def build_messages(user_question: str, context_str: str):
    system = (
        "Eres un asistente RAG. Responde ÚNICAMENTE usando el CONTEXTO proporcionado. "
        "Si la respuesta no está en el contexto, di explícitamente que no encuentras la "
        "información en los documentos. Sé preciso y, si procede, incluye una lista corta. "
        "No inventes ni alucines."
    )
    user = (
        f"Pregunta: {user_question}\n\n"
        f"--- CONTEXTO INICIA ---\n{context_str}\n--- CONTEXTO FIN ---\n\n"
        "Instrucciones:\n"
        "- Responde basándote solo en el contexto.\n"
        "- Si no hay datos suficientes, dilo claro.\n"
        "- Al final, si procede, incluye referencias como [1], [2]… según el bloque del contexto usado."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
