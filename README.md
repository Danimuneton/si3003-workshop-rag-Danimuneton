# Workshop RAG · SI3003 Inteligencia Artificial

- **Nombre:** Daniel Muñeton
- **Código estudiantil:** …
- **Usuario de GitHub:** Danimuneton
- **Fecha:** 22 de octubre de 2026
- **Entrega a:** jipadillab@eafit.edu.co

## Progreso

> Leyenda: ✅ código/respuesta listos en el repo · ⏳ requiere ejecutar en Colab
> con tu `NVIDIA_API_KEY` y pegar la evidencia real (salidas/capturas) en
> `RESPUESTAS.md`.

| ID | Tarea | Puntos | Hecha | Evidencia |
|---|---|---:|:---:|---|
| T1 | Cuántos nodos viven en cada capa | 2 | ✅ | RESPUESTAS.md · T1 |
| T2 | ef, k y el tamaño de la cola | 2 | ✅ | RESPUESTAS.md · T2 |
| T3 | Inserción frente a búsqueda | 2 | ✅ | RESPUESTAS.md · T3 |
| T4 | El invariante de FAISSVectorStore | 2 | ✅ | RESPUESTAS.md · T4 |
| T5 | El costo de la fuerza bruta | 2 | ✅ | RESPUESTAS.md · T5 |
| A1 | Ejecuta la app RAG de punta a punta | 4 | ⏳ | nb 02b (celda A1) → pegar salida |
| A2 | Tu propio corpus de documentos | 6 | ✅ | data/company_documents.json + nb 02b §3 |
| A3 | Ingeniería del prompt (antes y después) | 5 | ✅ | nb 02b §4 (PROMPT_ANTES/DESPUES) |
| A4 | Muestra el score de similitud en las fuentes | 5 | ✅ | nb 02b §5 (fuentes_con_score) |
| B1 | Ejecuta el RAG de papers y evalúa con el ground truth | 6 | ⏳ | nb 03 (QUIZ_B1) → pegar salida |
| B2 | Experimentos con chunk_size, k y score_threshold | 8 | ✅ | nb 03 (build_store/build_chain) + nb 04 (C4) |
| B3 | Preguntas trampa: ¿el RAG admite que no sabe? | 6 | ✅ | nb 03 (PREGUNTAS_TRAMPA) |
| B4 | Preguntas en español sobre papers en inglés | 5 | ✅ | nb 03 (PREGUNTAS_ES) |
| B5 | Escribe tu propio quiz con ground truth | 5 | ✅ | nb 03 (MI_QUIZ + BONUS) |
| R1 | Reflexión final con tus datos | 10 | ⏳ | RESPUESTAS.md · R1 → completar con tus números |
| C1 | HNSW contra fuerza bruta con FAISS (reto) | 15 | ✅ | nb 04 (C1, 3 TODO resueltos) |
| C2 | Un tercer paper: el RAG que explica HNSW (reto) | 12 | ✅ | nb 03 §1 (PAPERS: hnsw_paper) |
| C3 | App Gradio para los papers con filtro por paper (reto) | 15 | ✅ | nb 03 (demo_papers) |
| C4 | Evaluación automática de la recuperación (reto) | 12 | ✅ | eval/preguntas.csv (10 filas) + nb 04 (C4) |
| C5 | MMR frente a similitud pura (reto) | 8 | ✅ | nb 03 (build_chain search_type="mmr") |
| C6 | Persistencia del índice en disco (reto) | 8 | ✅ | nb 03 y nb 04 (QdrantClient path=...) |

**Autoevaluación:** … / 100 puntos (completa tras ejecutar y pegar las evidencias ⏳).

## Estructura

```
notebooks/
  02b_rag_app_nvidia_nim_colab.ipynb      # App RAG empresarial (A1-A4)
  03_rag_workshop_papers_nvidia_nim_colab.ipynb  # RAG de papers (B1-B5, C2, C3, C5, C6)
  04_challenges.ipynb                      # Retos C1, C4, C6
  04_challenges_starter.ipynb              # (original del curso)
data/
  company_documents.json                   # mi corpus (A2)
  company_documents.example.json           # ejemplo original
eval/
  preguntas.csv                            # 10 preguntas con ground truth (C4)
RESPUESTAS.md                              # respuestas y análisis
```

## Cómo ejecutar mis notebooks

1. Abre cada notebook de `notebooks/` en Google Colab.
2. Crea el secreto `NVIDIA_API_KEY` en Colab (panel Secrets) con acceso para el
   notebook. También puedes escribirla cuando `getpass` la pida.
3. Para cargar mi corpus y el CSV de evaluación, clona el repo dentro de Colab:
   `!git clone https://github.com/Danimuneton/si3003-workshop-rag-Danimuneton.git`
   y ejecuta desde esa carpeta, o sube `data/` y `eval/` a la sesión.
4. Ejecuta las celdas en orden. El notebook 04 (C4/C6) necesita variables del
   notebook 03 (`textos`, `build_store`, `all_docs`, …): ejecuta primero las
   partes 0–3 del notebook 03 en el mismo runtime.

> No incluyas tu API key en ningún archivo de este repositorio.

## Decisiones y aprendizajes

- Separé la lógica en funciones reutilizables (`build_pipeline`, `build_store`,
  `build_chain`) para poder variar `chunk_size`, `k`, `score_threshold` y
  `search_type` sin duplicar código — esto hizo B2, C4 y C5 casi triviales.
- El **prompt** resultó ser la palanca más importante: la única diferencia entre
  un RAG que alucina y uno que admite "no sé" fue instruirlo explícitamente
  (A3, B3).
- Mostrar el **score de similitud** (A4) ayuda a entender por qué el retriever
  trae o descarta un chunk, y a calibrar el `score_threshold`.
- La **persistencia en disco** (C6) y la evaluación automática con `hit@k` (C4)
  convierten el workshop de "probar a mano" a un flujo medible y repetible.
- HNSW (C1) solo compensa frente a la fuerza bruta cuando el corpus es grande;
  con pocos miles de vectores la búsqueda exacta ya es suficientemente rápida.
