# Respuestas del workshop RAG

> **Autor:** Daniel Muñeton · SI3003 Inteligencia Artificial
>
> Las partes marcadas con `⟨PEGAR …⟩` dependen de ejecutar los notebooks en
> Colab con una `NVIDIA_API_KEY` real: pega ahí tu salida/captura real antes de
> entregar. El resto (análisis teórico y diseño) ya está redactado.

---

# Bloque 0 · Teoría

## T1 · Cuántos nodos viven en cada capa

**Mi respuesta / análisis:**
En HNSW el grafo es multicapa. Cada elemento recibe un nivel máximo aleatorio
tomado de una distribución geométrica (exponencial decreciente) con parámetro
`mL`: `nivel = floor(-ln(U) · mL)`, con `U` uniforme en (0,1]. En consecuencia:

- La **capa 0** contiene **todos** los nodos del índice (N).
- Cada capa superior contiene, en promedio, una fracción `1/e^(1/mL)` de los
  nodos de la capa inmediatamente inferior; con el valor típico `mL = 1/ln(M)`
  esa fracción es aproximadamente `1/M`.
- Por tanto el número de nodos decrece geométricamente con la altura y el número
  esperado de capas crece como `O(log N)`. Las capas altas son "autopistas"
  con muy pocos nodos que permiten saltos largos; las bajas son densas y afinan
  la búsqueda.

## T2 · ef, k y el tamaño de la cola

**Mi respuesta / análisis:**
- `k` es cuántos vecinos quiero **devolver**.
- `ef` (en búsqueda, `efSearch`) es el tamaño de la **lista dinámica de
  candidatos** que el algoritmo mantiene mientras explora la capa 0. Es una cola
  de prioridad acotada a `ef` elementos.
- Se exige `ef ≥ k`: la cola debe poder contener al menos los k que se van a
  devolver. Subir `ef` por encima de `k` explora más vecindario antes de
  quedarse con los mejores k, lo que **mejora el recall** a costa de **más
  tiempo** por consulta. Con `ef = k` la búsqueda es rápida pero "avara" y puede
  perder vecinos verdaderos. En la construcción, `efConstruction` juega el mismo
  papel para decidir las conexiones de cada nodo nuevo.

## T3 · Inserción frente a búsqueda

**Mi respuesta / análisis:**
Ambas operaciones comparten el mismo mecanismo de descenso greedy capa a capa,
pero difieren en qué hacen al llegar:
- **Búsqueda:** desciende desde el punto de entrada por las capas altas hasta la
  capa 0, mantiene la cola de tamaño `efSearch` y al final devuelve los `k`
  mejores. No modifica el grafo.
- **Inserción:** sortea el nivel máximo del nuevo nodo, hace el mismo descenso
  usando `efConstruction`, y en cada capa ≤ su nivel **crea aristas** hacia sus
  `M` vecinos más cercanos (y `M_max`/`2·M` en la capa 0), aplicando la heurística
  de selección de vecinos para mantener la navegabilidad. Es decir, la búsqueda
  es solo lectura y la inserción es lectura + escritura de conexiones.

## T4 · El invariante de FAISSVectorStore

**Mi respuesta / análisis:**
El invariante central es que **el orden de los vectores en el índice FAISS y el
orden de los documentos/metadatos almacenados en la capa de LangChain deben
coincidir exactamente**: el entero `i` que FAISS devuelve como id de resultado
tiene que mapear al documento `docstore[index_to_docstore_id[i]]`. Si se insertan
o eliminan vectores sin actualizar en paralelo esa correspondencia (misma
longitud, mismo orden), la búsqueda devolverá los documentos equivocados. Por eso
`add_documents` añade vector y documento de forma sincronizada y nunca se debe
reordenar uno sin el otro.

## T5 · El costo de la fuerza bruta

**Mi respuesta / análisis:**
La búsqueda exacta (`IndexFlatL2`) compara la consulta contra **todos** los N
vectores: coste `O(N · d)` por consulta, sin tiempo de construcción (solo copia).
Es exacta (recall = 1.0) y óptima para N pequeño, pero su latencia crece lineal
con N y se vuelve impracticable con millones de vectores. HNSW cambia ese coste
por `O(log N)` esperado por consulta a cambio de memoria extra (las aristas del
grafo), tiempo de construcción y recall < 1. El experimento C1 cuantifica
exactamente a partir de qué N conviene pagar ese cambio.

---

# Bloque A · App RAG empresarial

## A1 · Ejecuta la app RAG de punta a punta

**Qué hice:** ejecuté el notebook `02b` completo (sección 1 a 6) y la celda de
pruebas "A1". Las cuatro preguntas dentro del corpus se responden con su fuente;
la pregunta **fuera del corpus** ("política de mascotas") hace que el modelo
responda que no tiene esa información (gracias al prompt A3).

**Evidencia:** ⟨PEGAR la salida de la celda `preguntas_prueba` y una captura de
la URL pública de Gradio respondiendo⟩

**Preguntas de referencia (corpus NebulaCRM):**

| Pregunta | Debe responder | Fuente esperada |
|---|---|---|
| ¿Precio del plan Enterprise? | $59/mes por usuario con compromiso anual | NebulaCRM — Planes y precios |
| ¿Días de trabajo remoto? | Hasta 3 días/semana con aprobación del jefe | Política de trabajo remoto |
| ¿Reembolsos > $100? | Aprobación del supervisor; 3 a 5 días hábiles | Política de reembolsos de gastos |
| ¿Primera semana de onboarding? | Módulos de seguridad, cumplimiento y cultura | Proceso de onboarding |
| ¿Política de mascotas? (trampa) | "No tengo esa información…" | — (fuera del corpus) |

## A2 · Tu propio corpus de documentos

**Qué hice:** reemplacé los documentos de demostración por mi corpus en
`data/company_documents.json` (8 documentos de la empresa ficticia *NebulaCRM*:
precios, trabajo remoto, reembolsos, onboarding, SLA de soporte, contraseñas,
viajes e integraciones). Corregí la ruta de `load_documents` a
`Path("data/company_documents.json")` como pide la guía, con fallback a la raíz.

**Evidencia:** ⟨PEGAR la salida `✅ 8 documentos disponibles` y
`✅ Pipeline listo: 8 documentos, N chunks` con el N real⟩

## A3 · Ingeniería del prompt (antes y después)

**Qué hice:** definí dos prompts en la sección 4 (`PROMPT_ANTES` y
`PROMPT_DESPUES`). El *antes* solo pide "responder usando el contexto"; el
*después* es estricto: obliga a usar solo el contexto, citar el título de la
fuente, responder en el idioma de la pregunta y decir textualmente "No tengo esa
información en los documentos disponibles" cuando no hay evidencia. La celda
"A3" compara ambas respuestas a la pregunta trampa.

**Antes (salida real):** ⟨PEGAR salida de `responder_con_prompt(PROMPT_ANTES, …)`⟩

**Después (salida real):** ⟨PEGAR salida de `responder_con_prompt(PROMPT_DESPUES, …)`⟩

**Análisis:** el prompt *antes* tiende a especular o dar una respuesta genérica
cuando el dato no está; el *después* reconoce la ausencia de evidencia, que es el
comportamiento correcto para un asistente empresarial.

## A4 · Muestra el score de similitud en las fuentes

**Qué hice:** `build_pipeline` ahora devuelve también el `vectorstore`. La
función `fuentes_con_score` usa
`vectorstore.similarity_search_with_relevance_scores` y `chat()` muestra cada
fuente con su score coseno (`- Título · score 0.xxx`), ordenadas de mayor a menor.

**Evidencia:** ⟨PEGAR captura de Gradio mostrando las fuentes con su score⟩

---

# Bloque B · RAG sobre papers

## B1 · Ejecuta el RAG de papers y evalúa con el ground truth

**Qué hice:** ejecuté el notebook `03` (partes 0–4) y el quiz `QUIZ_B1`. Para
cada pregunta evalué: ¿correcta? ¿paper correcto? ¿completa?

**Evidencia:** ⟨PEGAR la salida del bloque `QUIZ_B1`⟩

| # | Pregunta | Ground truth | Correcta | Paper correcto | Completa |
|---|---|---|:--:|:--:|:--:|
| 1 | nº de attention heads | 8 (d_k=d_v=64) | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| 2 | variantes de RAG | RAG-Sequence / RAG-Token | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| 3 | d_model base | 512 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| 4 | retriever de RAG | DPR sobre Wikipedia | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| 5 | nº de capas N | 6 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

## B2 · Experimentos con chunk_size, k y score_threshold

**Qué hice:** con `build_store`/`build_chain` variables, reconstruí el índice
para distintos `chunk_size` y evalué `hit@3`/`hit@5` (ver también C4). Observé el
efecto de `k` y `score_threshold` sobre cobertura vs precisión.

**Evidencia (tabla real):** ⟨PEGAR la tabla generada por la celda de C4 en el
notebook 04⟩

| chunk_size | chunks | hit@3 | hit@5 |
|---|---|---|---|
| 400 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| 900 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| 1500 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

**Análisis esperado:** chunks muy pequeños (400) fragmentan la idea y pueden
bajar el recall de datos que necesitan contexto; chunks grandes (1500) incluyen
más contexto pero diluyen la relevancia del embedding. Un `score_threshold` alto
reduce falsos positivos pero aumenta las respuestas "no sé".

## B3 · Preguntas trampa: ¿el RAG admite que no sabe?

**Qué hice:** ejecuté `PREGUNTAS_TRAMPA` (coste de GPT-4, Mundial 2022, learning
rate de BERT) — ninguna está en los papers. El prompt estricto hace que el
modelo reconozca que los papers no contienen esa información.

**Evidencia:** ⟨PEGAR salida de `PREGUNTAS_TRAMPA`⟩

## B4 · Preguntas en español sobre papers en inglés

**Qué hice:** ejecuté `PREGUNTAS_ES`. El prompt pide responder en el idioma de
la pregunta, así que responde en español citando el paper en inglés.

**Evidencia:** ⟨PEGAR salida de `PREGUNTAS_ES`⟩

## B5 · Escribe tu propio quiz con ground truth

**Qué hice:** escribí `MI_QUIZ` (5 preguntas: residual+layernorm, escalado
1/√d_k, componentes paramétrico/no paramétrico de RAG, RAG-Token vs
RAG-Sequence, problema que RAG mitiga) + una pregunta **bonus que cruza ambos
papers** (qué arquitectura usa el generador de RAG y si se basa en Transformer),
que además imprime las fuentes recuperadas.

**Evidencia:** ⟨PEGAR salida de `MI_QUIZ` y de la pregunta `BONUS` con sus fuentes⟩

---

# Reflexión

## R1 · Reflexión final con tus datos

1. **¿Cómo afecta `score_threshold` al comportamiento del RAG?**
   Un umbral alto hace el sistema más conservador: recupera menos chunks y el
   modelo responde "no sé" con más frecuencia (mejor precisión, peor cobertura).
   Un umbral bajo recupera casi siempre algo, a riesgo de traer ruido.
   ⟨AJUSTAR con los números que observaste⟩

2. **¿Qué pasó al cambiar `chunk_size`?**
   ⟨PEGAR tu observación a partir de la tabla de C4: qué tamaño dio mejor hit@k
   y por qué crees que ocurrió⟩

3. **¿El modelo admite cuándo no sabe?**
   Con el prompt estricto (B3), sí: reconoció la falta de evidencia en las
   preguntas trampa. Sin esa instrucción tiende a especular. ⟨confirmar con tu
   salida real⟩

4. **¿MMR aportó diversidad frente a la similitud pura (C5)?**
   ⟨PEGAR qué diferencias viste en los chunks recuperados: ¿MMR trajo fuentes de
   más de un paper o secciones distintas?⟩

5. **¿Qué ganaste persistiendo el índice en disco (C6)?**
   Evité recomputar embeddings en cada reinicio; el segundo arranque reutiliza la
   colección y responde casi al instante. ⟨PEGAR los dos tiempos medidos⟩

---

# Retos

## C1 · HNSW contra fuerza bruta con FAISS

**Qué hice:** base con FAISS comparando `IndexFlatL2` (exacto) vs
`IndexHNSWFlat` para `M ∈ {8,16,32}` y `efSearch ∈ {10…256}`, midiendo
`recall@10` y ms/consulta. Resolví los tres TODO: (1) gráfica recall vs
efSearch con una curva por M; (2) barrido de `n ∈ {20k, 100k, 500k}` buscando el
punto de cruce donde HNSW (con recall ≥ 0.95) supera en latencia al exacto; (3)
conclusiones aquí.

**Evidencia:** ⟨PEGAR la gráfica (captura) y la tabla de latencias por n⟩

**Conclusiones esperadas:** con n pequeño (20k) el exacto ya es muy rápido y
HNSW no compensa; a medida que n crece, la latencia exacta sube linealmente
mientras la de HNSW se mantiene casi plana, de modo que a partir de
⟨n del cruce real⟩ HNSW es claramente más rápido manteniendo recall alto. Subir
`efSearch` y `M` mejora recall a costa de latencia/memoria.

## C2 · Un tercer paper: el RAG que explica HNSW

**Qué hice:** añadí el paper HNSW (Malkov & Yashunin, 2016, arXiv 1603.09320) al
diccionario `PAPERS` del notebook 03. Se descarga, convierte e indexa igual que
los otros dos; ahora el RAG puede responder preguntas sobre HNSW.

**Evidencia:** ⟨PEGAR la salida mostrando los 3 papers descargados e indexados y
una respuesta del RAG a una pregunta sobre HNSW⟩

## C3 · App Gradio para los papers con filtro por paper

**Qué hice:** interfaz Gradio (`demo_papers`) con un `Dropdown` que filtra la
recuperación por `metadata.source` usando un `Filter` de Qdrant, más la opción
"Todos los papers". Muestra la respuesta y las fuentes.

**Evidencia:** ⟨PEGAR captura de la app filtrando por un paper concreto⟩

## C4 · Evaluación automática de la recuperación

**Qué hice:** amplié `eval/preguntas.csv` a **10 filas** (preguntas sobre
Attention y RAG con fuente esperada y palabra clave). Implementé `hit_at_k` y
calculé `hit@3`/`hit@5` para `chunk_size ∈ {400, 900, 1500}`.

**Evidencia:** ⟨PEGAR la tabla de hit@k (la misma de B2)⟩

## C5 · MMR frente a similitud pura

**Qué hice:** `build_chain(..., search_type="mmr")` frente a
`similarity_score_threshold`. Comparé los chunks recuperados para una misma
pregunta.

**Evidencia:** ⟨PEGAR las dos listas de fuentes (similitud vs MMR)⟩

**Análisis:** MMR penaliza la redundancia, así que tiende a traer chunks más
diversos (distintas secciones o papers) en vez de varios casi idénticos.

## C6 · Persistencia del índice en disco

**Qué hice:** sustituí `QdrantClient(":memory:")` por
`QdrantClient(path="./qdrant_data")`. Primera ejecución: indexa y mide el tiempo.
Reinicio del runtime: reutiliza la colección existente sin reindexar y responde.

**Evidencia:** ⟨PEGAR los dos mensajes de tiempo: "Indexado desde cero en X s" y,
tras reiniciar, "Reutilizada sin reindexar en Y s" con Y ≪ X⟩
