import requests
import streamlit as st

st.set_page_config(page_title="RAG Chroma – Front mínimo", layout="wide")
st.title("RAG ChatBot (Q&A sobre /answer)")

with st.sidebar:
    st.header("Ajustes API")
    base_url = st.text_input("URL base", value="http://127.0.0.1:8000")
    outdir = st.text_input("outdir", value="./chroma_store")
    collection = st.text_input("collection", value="rag_chunks")
    st.markdown("---")
    st.subheader("Parámetros LLM")
    model = st.text_input("Modelo Groq", value="llama-3.1-8b-instant")
    temperature = st.slider("temperature", 0.0, 1.0, 0.2, step=0.01)
    max_tokens = st.number_input(
        "max_tokens", min_value=64, max_value=4096, value=700, step=1
    )
    top_k = st.slider("top_k (recuperación)", 1, 20, 5, step=1)
    st.caption("Estos parámetros se envían a POST /answer")

st.subheader("Pregunta")
question = st.text_input("Escribe tu pregunta", value="¿Cuál es el objetivo del texto?")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    ask = st.button("Preguntar (POST /answer)", type="primary")
with col2:
    quick_search = st.button("Probar recuperación (POST /search)")
with col3:
    clear = st.button("Limpiar")

if clear:
    st.session_state.pop("last_answer", None)
    st.session_state.pop("last_search", None)


def post_answer():
    url = f"{base_url.rstrip('/')}/answer"
    body = {
        "outdir": outdir,
        "collection": collection,
        "query": question,
        "top_k": int(top_k),
        "model": model,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    r = requests.post(url, json=body, timeout=180)
    return r


def post_search():
    url = f"{base_url.rstrip('/')}/search"
    body = {
        "outdir": outdir,
        "collection": collection,
        "query": question,
        "topk": int(top_k),
    }
    r = requests.post(url, json=body, timeout=60)
    return r


if ask and question.strip():
    try:
        resp = post_answer()
        st.write("Status:", resp.status_code)
        if resp.status_code == 200 and "application/json" in resp.headers.get(
            "Content-Type", ""
        ):
            data = resp.json()
            st.session_state["last_answer"] = data
        else:
            st.error("Respuesta no-JSON o error del servidor.")
            st.code(resp.text)
    except Exception as e:
        st.error(f"Error llamando a /answer: {e}")

if quick_search and question.strip():
    try:
        resp = post_search()
        st.write("Status:", resp.status_code)
        if resp.status_code == 200 and "application/json" in resp.headers.get(
            "Content-Type", ""
        ):
            data = resp.json()
            st.session_state["last_search"] = data
        else:
            st.error("Respuesta no-JSON o error del servidor.")
            st.code(resp.text)
    except Exception as e:
        st.error(f"Error llamando a /search: {e}")

# ---- Render respuesta de /answer ----
ans = st.session_state.get("last_answer")
if ans:
    st.markdown("### Respuesta")
    st.write(ans.get("answer", "(sin contenido)"))

    st.markdown("#### Metadatos")
    meta_cols = st.columns(3)
    meta_cols[0].metric("Modelo", ans.get("used_model", "—"))
    meta_cols[1].metric("Longitud respuesta", len(ans.get("answer", "")))
    meta_cols[2].metric("Tiene fuentes", "Sí" if ans.get("sources") else "No")

    with st.expander("Fuentes"):
        sources = ans.get("sources") or []
        if sources:
            for s in sources:
                idx = s.get("idx")
                score = s.get("score")
                src = s.get("source") or ""
                page = s.get("page") or ""
                snippet = s.get("snippet") or ""
                st.markdown(f"**Fuente {idx}** — score: {score}")
                if src or page:
                    st.caption(f"{src} {('(p. '+str(page)+')' if page else '')}")
                if snippet:
                    st.info(snippet)
                st.markdown("---")
        else:
            st.write("(sin fuentes)")

    with st.expander("Respuesta cruda /answer (JSON)"):
        st.json(ans)

# ---- Render resultados de /search ----
srch = st.session_state.get("last_search")
if srch:
    st.markdown("### Resultados de /search (hits)")
    results = (srch or {}).get("results") or []
    if not results:
        st.write("(sin resultados)")
    else:
        for i, h in enumerate(results, 1):
            st.markdown(f"**Hit {i}** — score: {h.get('score')} — id: {h.get('id')}")
            st.info(
                (h.get("text") or "")[:900]
                + ("..." if len(h.get("text") or "") > 900 else "")
            )
            st.markdown("---")
