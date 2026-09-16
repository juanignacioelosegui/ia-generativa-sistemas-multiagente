# Laboratorio de Lenguaje Emergente — Informe

**Grupo:** <!-- número --> · **Integrantes:** <!-- nombres -->

> Plantilla. Cada `<!-- COMPLETAR -->` es de ustedes: la consigna dice explícitamente que las decisiones, la
> lectura de los resultados y las opiniones son la parte que se corrige, y que la escriban ustedes. Al lado de
> cada sección está de dónde sale cada número (los notebooks guardan todo en `checkpoints/`).

---

## 0 · Registro de decisiones

### Con qué contábamos al arrancar
<!-- COMPLETAR: horas reales del grupo, máquina (Colab T4 gratis / GPU local), cuánto rinde una sesión. -->

### Plan inicial
| Etapa | Plan (completa / mínima / se saltea) | Por qué |
|---|---|---|
| 1 · Tokenizador | | |
| 2 · Preentrenamiento + ablación | | |
| 3 · Embeddings | | |
| 4 · SFT (+ LoRA) | | |
| 5 · Juez | | |

### Bitácora
| Fecha | Qué probamos | Qué pasó | Qué decidimos y por qué |
|---|---|---|---|
| | | | |

### Plan contra realidad
<!-- COMPLETAR: dónde terminaron respecto del plan, y qué los hizo cambiar de idea. -->

### Cómo usamos IA
<!-- COMPLETAR: pocas líneas, como pide la §2 de la consigna. -->

---

## 1 · Etapa 0 — El dataset como instrumento
*Números:* `00_dataset.ipynb` (largo de cuentos, vocabulario aproximado).

<!-- COMPLETAR: ¿qué da experimentalmente un vocabulario de chico de 3–4 años que texto web no da?
     ¿Qué esperan que cambie con la misma cantidad de tokens de texto web, y por qué? -->

## 2 · Etapa 1 — Tokenización
*Números:* `01_tokenizer.ipynb`, `checkpoints/stage1_stats.json`, `checkpoints/stage1_vocab_tradeoff.png`.

| Tokenizador | V | tokens/cuento | chars/token |
|---|---|---|---|
| BPE propio | | | |
| GPT-2 (`r50k_base`) | 50.257 | | |
| `cl100k_base` | ~100K | | |

<!-- COMPLETAR: primeros merges y qué dicen del corpus; comparación con GPT-2; ¿vocab más chico siempre mejor? -->

## 3 · Etapa 2 — Preentrenamiento y ablación
*Números:* `02_pretraining.ipynb`, `checkpoints/stage2_ablation.json`, `stage2_loss_curves.png`,
`stage2_context_probe.png`.

**Configuración elegida y por qué:** <!-- COMPLETAR: block_size, batch, pasos, cantidad de datos. -->

**Hipótesis de la ablación:** <!-- COMPLETAR antes de mirar los resultados. -->

| | wide | deep |
|---|---|---|
| n_embd / n_layer | 256 / 2 | 192 / 5 |
| parámetros totales | | |
| parámetros fuera de embedding/lm_head | | |
| val loss final / perplejidad | | |
| recuerdo top-1 / top-5 | | |
| margen contexto (nombre) k=0 / k=6 | | |
| minutos / memoria pico | | |

**Chinchilla:** <!-- COMPLETAR: tokens vistos vs. ~20×params; qué creen que les costó. -->

<!-- COMPLETAR: ¿en qué rango de pérdida aparece estructura de oración? ¿Reproduce el paper? Si no, ¿por qué?
     ¿Qué recortaron y qué habrían medido con más? -->

## 4 · Etapa 3 — Representación aprendida
*Números:* `03_embeddings.ipynb`, `checkpoints/stage3_embeddings.json`, `stage3_pca.png`, `stage3_tsne.png`.

**Piso de ruido** (n_embd = , pool = palabras): <!-- COMPLETAR -->

| Par | Esperado | Coseno sin entrenar | Coseno entrenado | Percentil vs. control |
|---|---|---|---|---|
| | cerca | | | |
| | cerca | | | |
| | lejos | | | |

<!-- COMPLETAR: ¿le ganan al ruido? ¿Qué cambió en la proyección? ¿Filtrar a palabras enteras cambia algo? -->

## 5 · Etapa 4 — SFT
*Números:* `04_sft.ipynb`, `checkpoints/stage4_sft.json`, `stage4_sft_loss.png`.

**Par antes/después** (prompt: `Words: dragon, happy, forest`):

- Base: <!-- pegar -->
- SFT: <!-- pegar -->

| Modelo | palabras pedidas | control | empieza como cuento | cierra con EOS | loss TinyStories plano |
|---|---|---|---|---|---|
| base | | | | | |
| full FT | | | | | |
| LoRA | | | | | |

<!-- COMPLETAR: ¿qué cambió, forma o contenido? LoRA: fracción entrenada y cuánto de la mejora compró. -->

## 6 · Etapa 5 — Juez
*Números:* `05_judge.ipynb`, `checkpoints/stage5_judge.json`, `stage5_judge.png`.

<!-- COMPLETAR: ¿separa piso de techo? ¿Coincide con la perplejidad? ¿Las diferencias superan el ruido del juez? -->

## 7 · Aporte propio / conclusiones
<!-- COMPLETAR -->
