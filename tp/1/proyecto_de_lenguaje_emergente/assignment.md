# Proyecto · Laboratorio de Lenguaje Emergente

<!-- ---------------------------------------------------------------------------
     ENCABEZADO POR INSTANCIA. Editá las dos líneas de abajo cuando corras esto en
     otra materia o cuatrimestre. Dos de esos valores se repiten más abajo y hay que
     actualizarlos junto con ellos: el tamaño de grupo en la §6, y las tres semanas
     en el bullet del disco de Colab de la §6. Todo lo demás se sostiene solo.
     --------------------------------------------------------------------- -->

**IA Generativa Avanzada y Sistemas Multi-Agente · UTDT**

Grupos de hasta **4**. Se publica el **domingo 2026-09-06** y se entrega el **domingo 2026-09-27**.

<!-- --------------------------- fin del encabezado por instancia ------------------- -->

---

## 0 · La idea

Este proyecto reproduce, a escala mínima y enteramente en Colab, el ciclo de vida real de un LLM —
`tokenizador → preentrenamiento → representación aprendida → SFT` — y les pide que muestren, con datos
propios, por qué el lenguaje "emerge".

**La tesis:** *misma arquitectura, mejores datos → aparece el significado.* Ya vimos un Transformer a
nivel de caracteres entrenado sobre Shakespeare escupir texto con la pinta visual de un guion teatral
pero sin ningún sentido. Acá, el mismo modelo, entrenado sobre datos curados (TinyStories), produce
cuentos cortos coherentes. Esa diferencia es el primer pilar del preentrenamiento — los datos — hecho
visible en sus propios números.

Cada etapa reutiliza algo que ya construyeron. Acá no hay maquinaria nueva: son las mismas piezas,
apuntadas a mejores datos, en una escala donde lo pueden ver pasar:

`su propio tokenizador → coherencia que sale de los datos → estructura que aparece en los embeddings →
comportamiento que instala el SFT → todo eso medido por un juez`

## 1 · Datos duros (para que discutan con evidencia y no de memoria)

Fuente principal: Eldan & Li, *TinyStories: How Small Can Language Models Be and Still Speak Coherent
English?*, arXiv:2305.07759 (2023). Estas son las afirmaciones concretas y chequeables que conviene
citar en el informe:

- Modelos de **menos de 10M de parámetros** generan inglés coherente entrenados sobre TinyStories;
  modelos de unos 125M (GPT-Neo/GPT-2 small) entrenados sobre corpus generales casi nunca lo logran, ni
  siquiera con muchísimo más entrenamiento. Ese contraste es todo el punto.
- La referencia del paper original: arquitectura GPT-Neo, ventana 256, contexto 512, un tokenizador de
  GPT-Neo recortado a sus 10K tokens más frecuentes. Trabajos posteriores (y este proyecto) entrenan un
  **tokenizador BPE sobre el corpus mismo** — eso es lo que justifica que la Etapa 1 pida BPE.
- Las capacidades aparecen **por escalones**: la gramática la dominan los modelos chicos; la
  consistencia del contenido y la creatividad recién aparecen más arriba. **El ancho (dimensión del
  embedding) va de la mano del conocimiento factual; la profundidad (cantidad de capas) va de la mano de
  la comprensión del contexto y del largo alcance.** Eso es justamente lo que la ablación de la Etapa 2
  les pide chequear contra sus propios dos modelos.
- **TinyStories-Instruct** le pone a cada cuento, adelante, un *subconjunto* al azar y en *orden* al
  azar de hasta cuatro tipos de instrucción: `Features:`, `Words:`, `Summary:`, `Random sentence:` —
  confirmado muestreando el dataset, no asumido. Ese es el dataset de SFT de la Etapa 4.
- Cómo evalúa el paper: un LLM que corrige y le pone nota a gramática, creatividad, consistencia y
  obediencia a la instrucción, "como un docente corrigiendo a un alumno". La Etapa 5 replica eso
  localmente con Ollama + Qwen3.

**Datasets:** `roneneldan/TinyStories` (preentrenamiento) y `roneneldan/TinyStories-Instruct` (SFT).
Usen el Instruct de `roneneldan`, no el espejo `skeskinen/TinyStories-Instruct-hf` que citan algunas
referencias — ese rompe en varias configuraciones de Windows (un bug de archivo de lock del cargador
viejo cuando la ruta de clonado es profunda); si les pasa, miren el docstring de `data.py`. La versión
de `roneneldan` guarda una *línea* por fila y no un ejemplo por fila: `data.load_instruct_records`
reagrupa las líneas (cortando en `<|endoftext|>`) en bloques completos
`líneas de encabezado... \nStory: \n\n...texto del cuento`.

## 2 · Hasta dónde llegar lo deciden ustedes

**Arranquen midiendo con qué cuentan.** Antes de escribir una línea de código, saquen la cuenta de
cuánto tiempo tiene realmente el grupo y cuánto les rinde una sesión gratuita de Colab por corrida. Con
eso a la vista, decidan qué etapas van a llevar hasta el final, cuáles van a hacer mínimamente y cuáles
van a saltear. Anoten ese plan: es la primera entrada de su registro de decisiones, y después lo van a
comparar con lo que realmente pasó.

Acá hay más de lo que cualquier grupo va a poder terminar bien. Ese es el punto: **cómo encarar este
proyecto lo deciden ustedes.**

Lo que se evalúa es también **qué eligieron hacer y por qué**: qué llevaron hasta el final, qué
recortaron o hicieron mínimamente, cómo repartieron el tiempo y la máquina que tenían. Priorizar acá es
una habilidad profesional, no un premio consuelo.

Un proyecto que hace **todo por la mitad** saca menos que uno que hace **menos, pero hasta el fondo, y
explica por qué recortó lo que recortó.** Y un proyecto con menos etapas pero con una mirada propia —
que conecte lo que hicieron con algo más que sepan, o que ponga en duda un resultado en lugar de solo
contarlo — puede sacar más que una implementación impecable y completa que no tenga nada de su autor
adentro.

**Sobre usar IA.** Úsenla. Escriban código con ella, debugueen con ella, discutan con ella. Pero fíjense
lo que le hace a esta consigna: si un modelo saca la implementación en una tarde, la implementación ya
no es lo que distingue su trabajo. Lo que queda es lo que no puede hacer por ustedes: decidir qué valía
la pena construir, darse cuenta de si un resultado es creíble, ver eso que nadie les pidió que miraran y
bancar una opinión que puedan defender. Esa parte escríbanla ustedes; es la parte que leemos. En el
informe cuenten, en pocas líneas, cómo la usaron.

**Entregable: un registro de decisiones.** Junto con el o los notebooks y el informe, lleven un registro
al día: con qué contaban al arrancar, qué probaron, qué recortaron y por qué (tiempo, VRAM, cuánto
rendía conceptualmente). Terminar en otro lado del que habían planeado está perfecto y es lo esperable
— díganlo, y digan qué les hizo cambiar de idea.

**Todo lo que agreguen de más es bienvenido.** Si aparece algo más interesante que las etapas de abajo —
una comparación que no sugerimos, una falla que se merece un párrafo, una conexión con otro paper o con
otra materia — vayan por ahí, y cuéntennos. Un hilo del que vale la pena tirar, si quieren un punto de
partida: ¿cómo se compara la cantidad de datos con la que entrenaron contra el óptimo de cómputo para
un modelo de su tamaño (Chinchilla, [arXiv:2203.15556](https://arxiv.org/abs/2203.15556))? Saquen la
cuenta de dónde quedaron parados, y qué creen que les costó.

**Se las van a tener que arreglar solos, y es a propósito.** Algo se va a romper: un dataset que no
baja, una sesión de Colab que se cae en la mitad de una corrida, un `CUDA out of memory`, un resultado
que no reproduce. Eso no es un contratiempo del proyecto: es el proyecto. Acá no hay un camino correcto
escondido que tengan que adivinar, ni nadie que les vaya a destrabar el problema. Si una etapa no corre
como está planteada, encuéntrenle la vuelta: achiquen el subconjunto, cambien el experimento por otro
que conteste la misma pregunta, o cambien la pregunta por una que sí puedan contestar con lo que les
corre. Después cuenten en el registro qué se rompió, qué probaron y por qué terminaron donde
terminaron. Piénsenlo como un encargo de un cliente: al cliente "no se pudo" no le sirve. Le sirve algo
que funcione y, si no es exactamente lo que pidió, la razón por la que es esto y no aquello. Un desvío
bien explicado es una entrega; un casillero vacío no.

## 3 · Las cinco etapas

### Etapa 0 · El dataset como instrumento

Bajen TinyStories de HuggingFace, revísenlo (largo de los cuentos, vocabulario, un par de ejemplos) y
argumenten *en el informe* por qué un vocabulario chico a propósito lo convierte en un instrumento de
laboratorio: aísla la variable "datos" del ruido de la escala. Un encuadre corto, sin más código que el
de inspección.

```python
from data import load_tinystories  # wrapper fino sobre datasets.load_dataset

ds = load_tinystories(limit=2000)
print(ds)
print(ds[0]["text"])

lengths = [len(s.split()) for s in ds["text"]]  # contar palabras alcanza como aproximación acá
```

**Preguntas para el informe:** ¿Qué les da experimentalmente "el vocabulario de un chico de 3–4 años"
que un corpus general de texto web no les da? Si entrenaran la misma arquitectura sobre un pedazo de
texto crudo de internet con la misma *cantidad de tokens*, ¿qué esperarían que cambie, y por qué?

---

### Etapa 1 · Tokenización — su propio BPE *(obligatoria)*

**Por qué esta etapa.** Un tokenizador no es un default que uno hereda: es algo que se entrena sobre
*sus* datos, y el tamaño del vocabulario se paga contra el largo de secuencia — el toma y daca `V ↔ T`.
Acá lo ven sobre datos que eligieron ustedes.

**Pasos sugeridos.**
1. Entrenen un tokenizador BPE a nivel byte sobre un subconjunto de TinyStories, apuntando a un
   vocabulario de 4.096–8.192, con `tokenizers` de HF (la librería de bajo nivel, no `transformers` —
   acá no hay ningún modelo para cargar).
2. Codifiquen y decodifiquen un par de cuentos; chequeen que el ida y vuelta dé exacto.
3. Miren los primeros ~20 merges que aprendió su tokenizador. ¿Qué capturan, y por qué *este* corpus
   produce *estos* merges primero y no, digamos, un corpus de código?
4. Tokenicen el mismo cuento con su BPE y con el tokenizador posta de GPT-2 — `tiktoken`, encoding
   `gpt2` (también conocido como `r50k_base`, 50.257 tokens). Comparen la cantidad de tokens. ¿Qué dice
   esa diferencia sobre el toma y daca `V ↔ T` en un corpus de vocabulario angosto? *(`cl100k_base` es
   un encoding posterior, de ~100K tokens, que usan GPT-3.5/GPT-4 — sirve como tercer dato si lo
   quieren, pero no es el de GPT-2.)* Ojo que `tiktoken` baja el encoding la primera vez que se usa, así
   que este paso necesita red.

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=8192,
    special_tokens=["<|endoftext|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)  # corpus_iterator: un iterable de strings

enc = tokenizer.encode(some_story)
print(enc.tokens[:20])
```

**Preguntas para el informe:** ¿Qué merges aparecieron primero, y qué les dicen sobre cómo está armado
este corpus? Con el vocabulario que eligieron, ¿cuántos tokens por cuento les da en promedio, y cómo se
compara con el tokenizador de GPT-2 sobre los mismos cuentos? ¿Un vocabulario más chico es siempre
mejor acá, o hay algo que están pagando a cambio?

---

### Etapa 2 · Preentrenamiento — el corazón del laboratorio

**Por qué esta etapa.** Esta etapa no se trata de armar un Transformer. Si ya armaron uno — pasando por
bigrama → self-attention → bloque completo — entonces esto es: **agarrar ese mismo código**, apuntar su
tabla de embeddings al vocabulario BPE de la Etapa 1 en lugar de a caracteres, y entrenarlo sobre
TinyStories. Si no lo tienen a mano, escríbanlo a partir del fragmento de abajo; en cualquier caso la
arquitectura es la constante. El ciclo `forward → loss → backward → update` es el de siempre, sin
cambios. Lo nuevo son los datos, y ese es todo el punto.

**Pasos sugeridos.**
1. Reutilicen el `GPTLanguageModel` que ya armaron (`Head` → `MultiHeadAttention` → `FeedForward` →
   `Block`, apilados, con el `LayerNorm` final y el `lm_head`). Cambien `vocab_size` de la cantidad de
   caracteres al tamaño del vocabulario de su tokenizador de la Etapa 1. Todo lo demás de la clase queda
   igual. *(¿No lo tienen a mano? Con la lista de componentes de arriba y el fragmento de abajo alcanza
   para escribirlo de cero: es un Transformer decoder-only estándar y rehacerlo es una tarde de trabajo.
   Eso sí, no vayan a agarrar `transformers`: en este pipeline no hay ningún checkpoint preentrenado, y
   la gracia de la etapa es que la arquitectura queda fija mientras cambian los datos.)*
2. Elijan una configuración dentro de los rangos de la §7 y entrenen sobre TinyStories, tokenizado con
   su tokenizador de la Etapa 1. Vayan siguiendo la pérdida en train y en held-out, y la perplejidad.
3. Generen muestras del modelo en varios puntos del entrenamiento (por ejemplo el paso 0, y algunos
   puntos en el medio). Léanlas: ¿en qué momento se estabiliza la gramática? ¿Y la consistencia a lo
   largo de un cuento entero? (Opcional: ¿coincide con lo que dice el paper, que la gramática aparece
   antes que la consistencia?)
4. **La ablación — ancho contra profundidad.** *(Bajarse de la Etapa 2 entera es una decisión válida
   bajo la §2; bajarse de la ablación pero dejar la corrida de entrenamiento es el único recorte que les
   vamos a discutir, así que si hacen eso defiéndanlo en el registro de decisiones.)* Entrenen dos
   modelos con una cantidad de parámetros parecida: uno **ancho** (`n_embd` más grande, menos capas) y
   uno **profundo** (`n_embd` más chico, más capas — el par está en la §7, y antes de elegir uno propio
   lean la trampa de emparejar parámetros que está más abajo). Compárenlos contra lo
   que encontró el paper (ancho↔conocimiento, profundidad↔contexto) usando pérdida y perplejidad,
   muestras cualitativas **y** un set chico y fijo de prompts de prueba que diseñen ustedes, partido en
   dos tipos: prompts que prueben si recuerda una palabra o un dato común (por ejemplo completar "the
   sun is ___"), y prompts que prueben si un detalle que apareció temprano en el cuento generado (un
   nombre, un objeto) se sigue usando bien varias oraciones después. Esta ablación, y no la corrida de
   entrenamiento base, es lo que convierte esto en un *laboratorio*.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ self.value(x)

# ... MultiHeadAttention / FeedForward / Block / GPTLanguageModel: las mismas formas que la versión que
# ya armaron, solo que con vocab_size = len(your_bpe_tokenizer.get_vocab()) en vez de la cantidad de caracteres.
```

**Trampa: "cantidad de parámetros parecida" es una restricción que se imponen ustedes, no algo que
venga de arriba.** Partir el ancho al medio y duplicar la profundidad *no* deja el tamaño fijo: lo
achica más o menos a la mitad. Con este vocabulario, la mayor parte del modelo es la tabla de
embeddings de tokens más la capa de salida, y las dos son `O(vocab × n_embd)`: bajar `n_embd` achica el
grueso del modelo, mientras que sumar capas solo hace crecer lo que queda. Cuenten los parámetros de
los dos modelos antes de entrenar ninguno. Si difieren bastante ya no están corriendo un experimento de
ancho contra profundidad, están corriendo uno de grande contra chico, y el que gane no les dice nada
sobre la forma. Informen las dos cuentas.

**Trampa: sembrar la semilla adentro del loop de generación.** Como la reproducibilidad dice "fijen
todas las semillas", sale natural llamar a `torch.manual_seed(1337)` justo antes de generar desde cada
prompt de prueba. No lo hagan. Ahí cada generación saca los *mismos* números al azar, y en un modelo
poco entrenado ese ruido de muestreo puede tapar al prompt: prompts distintos vuelven con
continuaciones sospechosamente parecidas, y dos modelos genuinamente distintos parecen iguales. Siembren
una vez **por prompt** (`seed + i`), así un mismo prompt queda comparable entre modelos y prompts
distintos se separan. En la misma línea: para los prompts de recuerdo conviene decodificación greedy.
"¿Elige una palabra plausible?" es una pregunta sobre la predicción más probable del modelo, no sobre
una muestra a temperatura 1.0.

**Preguntas para el informe:** ¿En qué rango de pérdida el texto deja de ser puro ruido y empieza a
tener estructura de oración? ¿*Su* par ancho-profundo reproduce lo que encontró el paper o no, y si no,
cuál es su mejor explicación (escala, cuánto entrenaron, cómo diseñaron los prompts de prueba)? ¿Qué
tuvieron que recortar para que la ablación entrara en la máquina que tenían, y qué habrían medido con
más?

---

### Etapa 3 · La representación aprendida

**Por qué esta etapa.** Van a usar las mismas herramientas de similitud coseno que ya usaron sobre
vectores de palabras, pero sobre una matriz que entrenaron *ustedes* y no una preentrenada que bajaron
de algún lado. Y como sus embeddings viven sobre **tokens BPE**, y no sobre caracteres ni palabras
enteras de diccionario, esta etapa además cierra el círculo con la Etapa 1.

**Pasos sugeridos.**
1. Saquen la tabla de embeddings de tokens de su mejor checkpoint de la Etapa 2 (su primera capa).
   Guarden también una copia de un modelo **sin entrenar** (paso 0, misma semilla) como referencia.
2. Filtren a los tokens que son palabras enteras. El BPE a nivel byte marca el comienzo de palabra con
   un espacio adelante, pero ese espacio lo codifica como **`Ġ` (U+0120)** y no como un espacio ASCII,
   así que el token de `" dog"` es el string `"Ġdog"`. Filtren con `tok.startswith("Ġ")`; filtrar con
   `" "` no matchea nada y los deja, calladito, con una lista vacía. Para un par de estos (por ejemplo
   `Ġdog`/`Ġcat`, `Ġhappy`/`Ġsad`), calculen similitud coseno y vecinos más cercanos, igual que harían
   con cualquier espacio de vectores de palabras.
3. Proyecten una muestra de embeddings a 2D (PCA o t-SNE) y grafiquen el modelo sin entrenar y el
   entrenado, uno al lado del otro.
4. Comparen lo que ven: ¿el espacio entrenado deja más cerca a los tokens de palabras enteras que están
   relacionados por significado, comparado con el no entrenado?

```python
import numpy as np

# Achiquen PRIMERO el conjunto de candidatos a tokens de palabras enteras, y recién ahí ranqueen adentro. Si
# ranquean sobre todo el vocabulario, ~3/4 de cada lista de vecinos vuelve como pedazos de bytes y puntuación.
vocab = tokenizer.get_vocab()                       # {token_string: id}
whole_word = sorted((t, i) for t, i in vocab.items() if t.startswith("Ġ") and t[1:].isalpha())
words = [t[1:] for t, _ in whole_word]              # "Ġdog" -> "dog", para que las búsquedas se lean bien
rows = np.array([i for _, i in whole_word])         # las filas de embedding a las que apuntan esos tokens

# Normalicen una sola vez y cada coseno es un solo producto matriz-vector. Token por token en un loop de
# Python es ~V veces más lento, algo que van a sentir con vocabulario 8.192.
pool = embedding_matrix[rows]                       # embedding_matrix: su tabla de (vocab_size, n_embd)
unit = pool / np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)

def nearest(word, topn=5):
    if word not in words:
        raise KeyError(f"{word!r} no es un token de palabra entera en este vocabulario")
    q = words.index(word)
    sims = unit @ unit[q]
    return [(words[i], float(sims[i])) for i in np.argsort(-sims) if i != q][:topn]
```

**Trampa: muchas palabras están en el vocabulario dos veces.** Una como token de palabra entera
`"Ġday"` y otra como pedazo en el medio de una palabra `"day"` (como en `"birthday"`). El pedazo es otra
fila, entrenada sobre muchísimas menos apariciones, y con vocabulario 8.192 varias decenas de filas de
pedazos nunca aparecen durante el entrenamiento, así que se quedan con la inicialización al azar. Si
indexan por el string pelado les va a devolver, calladito, esa fila y una lista de vecinos sin ningún
sentido, sin tirar error. Armar el mapa a partir de los tokens con `Ġ`, como arriba, es lo que lo evita.

**Calculen su piso de ruido antes de afirmar nada.** La similitud coseno entre vectores *al azar* no da
cero, da chico: en `n_embd` dimensiones va como `1/√n_embd` — unos `0,06` con `n_embd=256` — y el valor
más grande entre todo un vocabulario de candidatos da varias veces eso. Así que una similitud de 0,2
puede no ser más que la mejor de varios miles de tiradas de moneda. Calculen ese piso para su propio
`n_embd` y su vocabulario, y traten todo lo que quede por debajo como no-evidencia. La misma lógica vale
para la comparación antes/después: que un par se acerque después de entrenar no dice mucho por sí solo,
así que agreguen un grupo de control de pares elegidos al azar y fíjense si los pares que esperaban ver
cerca se movieron *más* que esos. Un resultado flojo que pueden acotar vale más que uno fuerte que no.

**Preguntas para el informe:** Elijan dos pares de palabras que esperaban ver cerca y uno que esperaban
ver lejos — ¿le pegaron, y la diferencia le gana a su piso de ruido? ¿Qué cambió a la vista entre la
proyección 2D del modelo sin entrenar y la del entrenado? ¿Quedarse solo con los tokens de palabras
enteras cambia lo que ven, comparado con incluir los pedazos sub-palabra? Si el espacio entrenado
termina pareciéndose bastante al no entrenado, ¿qué cambiarían para moverlo, y qué les costaría?

---

### Etapa 4 · SFT — comportamiento, no conocimiento

**Por qué esta etapa.** La misma pérdida de próximo token que en el preentrenamiento; lo único que
cambia, una vez, son los datos, que pasan a ser demostraciones instrucción→cuento. Esa es toda la tesis
del fine-tuning, sobre un modelo que entrenaron ustedes.

**Pasos sugeridos.**
1. Sigan entrenando su checkpoint de la Etapa 2 sobre TinyStories-Instruct, usando
   `data.load_instruct_records` para tener los bloques de texto completos: un *subconjunto* al azar, en
   *orden* al azar, de las líneas `Features:`/`Words:`/`Summary:`/`Random sentence:`, seguidas de
   `Story:` y el cuento. Es un formato de continuación pelado: acá no hay estructura de chat ni de
   turnos, a diferencia de un SFT conversacional.
2. **El experimento que importa:** métanle el *mismo* prompt (por ejemplo `Words: dragon, happy,
   forest`) a su checkpoint base (solo preentrenado) y al de SFT. El base debería irse por las ramas e
   ignorar las palabras; el de SFT debería meterlas en el cuento. Esa diferencia *es* la distinción
   entre conocimiento y comportamiento, en sus propios modelos.
3. **Opcional — LoRA.** En vez de `peft` (acá no hay ningún modelo de HuggingFace donde engancharlo),
   escriban LoRA a mano: una `LoRALayer` chiquita (dos matrices de rango bajo `A` y `B`, con `B`
   arrancando en cero para que el adaptador al principio no haga nada) envolviendo las proyecciones de
   atención de su modelo (`Head.key`/`query`/`value`, `MultiHeadAttention.proj`). Comparen qué fracción
   de parámetros terminan entrenando, cuánta memoria usan y cómo sale la generación, contra el
   fine-tuning completo.

```python
import math

class LoRALayer(nn.Module):
    """Aprende ΔW ≈ (alpha/r) · B · A, con A al azar y chica y B arrancando en cero, así ΔW=0 al principio."""
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std = 1.0 / math.sqrt(rank)
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.scaling * (x @ self.A @ self.B)

# Envolver un nn.Linear que ya existe: congelar `linear` y entrenar solo la LoRALayer.
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False
        # El `.to()` importa: al modelo lo envuelven DESPUÉS de mandarlo a la GPU, y un
        # nn.Parameter nuevo nace en la CPU. Sin eso, el primer forward tira
        # "Expected all tensors to be on the same device" -- y solo en GPU, así que un
        # smoke test en CPU no se los va a agarrar.
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha).to(
            device=linear.weight.device, dtype=linear.weight.dtype
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

**Lean el par antes/después con atención: el comportamiento tiene más de una dimensión.** El SFT puede
cambiar *cómo* escribe el modelo (¿arranca un cuento cuando le pidieron un cuento?) por separado de
*qué* escribe (¿usa las palabras que le dieron?). Esas dos cosas se pueden mover por su cuenta, y con
poco entrenamiento puede pasar que se mueva una y la otra no. Un resultado parcial o negativo acá es un
hallazgo, no un fracaso: digan con precisión qué parte cambió, y qué harían para conseguir el resto.

**Preguntas para el informe:** Muestren en el informe el par antes/después con el mismo prompt — ¿qué
cambió concretamente, la forma o el contenido? Si hicieron LoRA: ¿qué fracción de los parámetros
entrenaron realmente, y cuánto de la mejora del fine-tuning completo les compró eso?

---

### Etapa 5 · Medir la emergencia *(opcional, recomendada)*

**Por qué esta etapa.** Un número de perplejidad no les dice *qué* mejoró. Acá replican la forma de
evaluar del propio paper con un juez local y gratis, y se llevan una segunda lectura, independiente, de
todo lo que hicieron en las Etapas 2 y 4.

**Pasos sugeridos.**
1. Levanten Ollama + `qwen3:4b` dentro de su sesión de Colab (`think=False`, para que puntúe directo y
   rápido: acá no hace falta el modo de razonamiento del modelo).
2. Que le ponga nota a un puñado de generaciones de su modelo base, su modelo con SFT y las dos
   variantes de la ablación, en gramática, creatividad, consistencia y —para las salidas con SFT—
   obediencia a la instrucción.
3. Comparen la lectura del juez contra la perplejidad y los prompts de prueba de la Etapa 2. ¿Coinciden?

```python
import subprocess, time, requests, ollama

subprocess.Popen(["ollama", "serve"])
for _ in range(30):
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        break
    except requests.exceptions.RequestException:
        time.sleep(1)
# después: !ollama pull qwen3:4b

response = ollama.chat(
    model="qwen3:4b",
    messages=[{"role": "user", "content": grading_prompt}],  # pedir gramática/creatividad/consistencia/obediencia, 1-10 cada una
    format="json",   # obliga al decoder a devolver JSON válido -- háganlo, ver abajo
    think=False,
)
```

**Trampa: a veces el juez contesta dos veces.** A un modelo chico al que le piden JSON cada tanto le da
por devolver dos objetos, o por envolver la respuesta en una oración. De ahí salen dos cosas. Primero,
el `format="json"` de arriba vale la línea que ocupa. Segundo, no lo extraigan con un
`re.search(r"\{.*\}", text, re.DOTALL)` glotón: eso agarra desde la *primera* llave hasta la *última*,
así que dos objetos vuelven como un solo string y `json.loads` tira `Extra data`. En cambio,
`json.JSONDecoder().raw_decode(text, i)` corta al final del primer valor. Y envuelvan cada llamada para
que una respuesta mala no les tire abajo la corrida entera: un juez que no respeta el formato es, en sí
mismo, un resultado que pueden informar.

**Preguntas para el informe:** ¿Dónde le da la razón el juez a la perplejidad, y dónde se le va para
otro lado? Cuando se le va para otro lado, ¿qué les dice eso sobre lo que la perplejidad mide y lo que
no?

## 3b · Qué produce cada etapa y qué le pasa a la siguiente

Cada etapa consume lo que dejó la anterior, así que el grupo tiene que ponerse de acuerdo en los nombres
de archivo antes de que nadie empiece — sobre todo si se reparten las etapas entre integrantes. Usen
estos; las rutas son relativas a la raíz del repositorio, y `checkpoints/` está en el gitignore, así que
nada pesado termina en la entrega.

| Archivo | Lo escribe | Lo lee |
|---|---|---|
| `checkpoints/tokenizer.json` | Etapa 1 (`tokenizer.save(...)`) | Etapas 2, 3, 4, 5 |
| `checkpoints/pretrain_step0.pt` | Etapa 2, antes de entrenar nada | Etapa 3 (la referencia sin entrenar) |
| `checkpoints/pretrain_wide.pt` | Etapa 2 | Etapas 3, 4, 5 |
| `checkpoints/pretrain_deep.pt` | Etapa 2 (ablación) | Etapa 5 |
| `checkpoints/sft_final.pt` | Etapa 4 | Etapa 5 |

Un checkpoint tiene que guardar lo suficiente como para reconstruir el modelo sin adivinar: el
`state_dict` **más** la configuración que define su forma (`vocab_size`, `n_embd`, `n_layer`, `n_head`,
`block_size`). Guardar solo los tensores es la forma más común de perder una corrida de entrenamiento.

**Lo que no se puede saltear**, para que sepan qué les cuesta cada recorte: la Etapa 3 necesita el
checkpoint del paso 0 *y* el entrenado de la Etapa 2; la Etapa 4 necesita el entrenado de la Etapa 2; la
Etapa 5 necesita la 2 y la 4. La Etapa 1 alimenta a todas. Recorten desde el final, no desde el medio.

## 4 · Qué entregar

- **El o los notebooks**, que corran en Colab, cubriendo las etapas que eligieron priorizar.
- **Un informe** (en markdown dentro del notebook o como documento aparte): decisiones de diseño y por
  qué las tomaron, curvas de pérdida y perplejidad, muestras, la tabla de la ablación ancho/profundidad,
  el análisis de embeddings, la comparación base contra SFT, y las respuestas a las preguntas
  conceptuales de cada etapa.
- **El registro de decisiones** (§2): qué intentaron, qué recortaron y por qué.

## 5 · Cómo se corrige

| Criterio | Qué miramos |
|---|---|
| **Priorización** | Qué tan bien recortaron con una restricción real encima; si cierra lo que dijeron que tenían al arrancar, lo que eligieron hacer y lo que cuenta el registro de decisiones. |
| **Diseño experimental** | La hipótesis de la ablación, qué dejaron fijo y qué variaron, cómo lo midieron. |
| **Comprensión conceptual** | Por qué está cada pieza y cómo se encadena el pipeline (tokenizador → preentrenamiento → embeddings → SFT). |
| **Lectura de los resultados** | Interpretar curvas, muestras, geometría de embeddings y —si lo corrieron— las notas del juez. No alcanza con mostrarlos. |
| **Aporte propio** | Análisis, experimentos o conexiones que nadie les pidió; una opinión que argumenten en vez de reportar. |
| **Que se pueda reproducir y se entienda** | El notebook corre; el informe se lee. |

**Tipear código no suma puntos.** Un proyecto con menos etapas pero bien pensado le gana a uno que las
hace todas por la mitad — y un proyecto con algo propio para decir les gana a los dos.

## 6 · Restricciones y logística

- **Todo código abierto**, nada de APIs pagas: `tokenizers` / `datasets` de HF, PyTorch, Ollama + Qwen3
  (el juez). En ningún punto del pipeline hace falta `transformers`, `peft`, `bitsandbytes` ni `trl`:
  todos los modelos de acá están hechos a mano, ninguno es un checkpoint de HuggingFace.
- **Dónde corre:** Google Colab, GPU T4 (versión gratuita). Usen subconjuntos y el flag `SMOKE_TEST`
  para iterar rápido antes de largarse a una corrida completa.
- **El disco de Colab se borra: cuenten con eso.** Cuando se desconecta el entorno de ejecución (más o
  menos 90 minutos de inactividad, 12 horas como máximo en la versión gratuita) desaparece todo lo que
  esté en `checkpoints/`, y las Etapas 3 a 5 dependen todas de los checkpoints de la Etapa 2. Este
  proyecto es de tres semanas; no lo van a hacer de una sentada. O montan Drive y dejan `checkpoints/`
  ahí (la celda de arranque de cada notebook tiene un bloque comentado que hace justamente eso), o bajan
  los archivos antes de cerrar la pestaña. Perder un modelo entrenado por una desconexión es la forma
  más común en que un grupo pierde un día.
- **Cómo llevar el código a Colab:** la primera celda de cada notebook clona el repositorio, entra con
  `cd` e instala `requirements.txt`. Córranla primero; hasta que no lo hagan no importa nada.
- **Datos:** TinyStories y TinyStories-Instruct, de HuggingFace.
- **Grupos:** hasta 4.

## 7 · Guía de tamaños

Rangos para arrancar. Elijan valores concretos que les entren; esa elección también es parte de lo que
se corrige. Son puntos de partida, no objetivos.

- **Tokenizador:** vocabulario 4.096–8.192, entrenado sobre un subconjunto del corpus.
- **Modelo:** `n_layer` 2–5, `n_embd` 128–256, `n_head` 4–8, `block_size` (contexto) 128–256, `dropout`
  0,1 → más o menos 1–10M de parámetros. Elijan `block_size` mirando los largos de cuento que midieron
  en la Etapa 0: cuánto de un cuento típico entra en una ventana de contexto, y si eso importa para lo
  que le están pidiendo aprender al modelo.
- **Par de la ablación:** ancho (`n_embd=256`, `n_layer=2`) contra profundo (`n_embd=192`, `n_layer=5`)
  — con vocabulario 8.192 y `block_size` 192 quedan a menos del 7% uno del otro. Cuenten los suyos en
  vez de confiar en los nuestros, y si van a elegir otro par lean antes la trampa de emparejar
  parámetros de la §Etapa 2.
- **Entrenamiento:** AdamW, `lr` ≈ 3e-4, batch 32–64, unos miles de pasos sobre un subconjunto.
- **SFT:** seguir desde el checkpoint base sobre un subconjunto de Instruct, menos pasos que en el
  preentrenamiento y con un `lr` más bajo (≈1e-4: están empujando un poquito un modelo ya entrenado, no
  arrancando de cero). LoRA (hecho a mano, §Etapa 4) es opcional.
- **El juez:** Ollama + `qwen3:4b`, `think=False`, `format="json"`. Puntúen varias generaciones por
  modelo y no una sola; con una muestra no van a poder separar dos modelos a esta escala.
- **`SMOKE_TEST = True`:** un subconjunto mínimo y ~200 pasos, para validar el pipeline en minutos antes
  de largarse a una corrida completa. Sus números no significan nada: existe para probar que el código
  corre.
- **Un chequeo de cordura que no es un objetivo.** La entropía cruzada arranca cerca de
  `ln(vocab_size)` — que es lo que cuesta adivinar uniformemente sobre el vocabulario, ≈9,0 con
  vocabulario 8.192. Si en unos cientos de pasos su pérdida no bajó claramente de ahí, el bug está más
  arriba: revisen la tokenización y los batches antes de tocar hiperparámetros.
- **Reproducibilidad:** `torch.manual_seed(1337)` en todos lados donde puedan — pero lean la trampa de
  la semilla de la §Etapa 2 antes de meterlo adentro de un loop.
