# ProyectoFinalAPI_IA

# 🤖 Chatbot Temático con RAG (Retrieval-Augmented Generation)

## 🎯 Objetivo
Desarrollar un chatbot capaz de responder preguntas basadas únicamente en un conjunto de documentos propios (PDF o TXT), utilizando técnicas de **búsqueda semántica** y **modelos de lenguaje (LLM)**.

---

## ⚙️ Funcionamiento General

El sistema se divide en **dos fases principales** y dos endpoints:

### 🟩 1) Ingesta de Documentos → `POST /documents`

1. **Entrada:**  
   El usuario sube uno o varios archivos PDF o TXT.

2. **Procesamiento:**  
   - Se extrae el texto de cada documento (añadiendo OCR si es necesario).  
   - El texto se divide en **fragmentos (chunks)**.  
   - A cada fragmento se le generan **embeddings**, que son vectores numéricos que representan el significado del texto.

3. **Almacenamiento:**  
   - Los embeddings se guardan en una **base vectorial (Chroma)** junto con metadatos como:  
     `doc_id`, `título`, `número de página`, etc.  
   - Esta base vectorial servirá para encontrar los fragmentos más relevantes durante las consultas.

---

### 🟦 2) Consulta del Usuario → `POST /chat/query`

1. **Entrada:**  
   El usuario escribe una pregunta.

2. **Procesamiento:**  
   - Se genera el **embedding de la pregunta**.  
   - Se realiza una **búsqueda semántica** en la base vectorial (por **similitud coseno**) para recuperar los **Top-K fragmentos** más relacionados.

3. **Generación de Respuesta:**  
   - Se construye un **prompt** con la pregunta y los fragmentos recuperados.  
   - Se envía el prompt a un **modelo de lenguaje (LLM)**, que puede ejecutarse localmente (Ollama) o por API (Groq, Mistral, Replicate, etc.).  
   - El modelo genera una respuesta **citando las fuentes** (documento y página).

4. **Salida:**  
   - Si hay resultados: ✅ Respuesta + citas.  
   - Si no hay coincidencias (K = 0): ⚠️ “No encontrado”.

---

## 🧩 Componentes Principales

- **FastAPI:** Framework backend para crear los endpoints.  
- **ChromaDB:** Base de datos vectorial donde se almacenan los embeddings.  
- **Sentence Transformers:** Librería para generar embeddings.  
- **LLM (Groq / Mistral / Ollama / Replicate):** Modelo de lenguaje que genera las respuestas.  
- **PyMuPDF:** Librería para extraer texto de los PDFs.  
- **(Opcional) OCR:** Para leer PDFs escaneados o imágenes.

---

## 🧱 Arquitectura Resumida

```text
📄 PDF / TXT
   ↓
[Extracción de texto + Chunking]
   ↓
[Embeddings (vectores)]
   ↓
🧠 Base Vectorial (Chroma)
   ↓
❓ Pregunta del Usuario
   ↓
[Embedding de la Pregunta]
   ↓
[Top-K Fragmentos más Similares]
   ↓
[Prompt con Contexto + Pregunta]
   ↓
🤖 LLM (Groq / Mistral / Ollama)
   ↓
✅ Respuesta + Citas
