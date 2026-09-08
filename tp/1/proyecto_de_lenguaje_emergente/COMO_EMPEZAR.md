# Proyecto · Laboratorio de Lenguaje Emergente

**IA Generativa Avanzada y Sistemas Multi-Agente · Licenciatura en Tecnología Digital · UTDT**

Grupos de **4** personas. Se entrega el **domingo 2026-09-27** por el campus.

---

## Qué es esto

Van a reproducir, a escala mínima y enteramente en Colab, el ciclo de vida completo de un LLM:
`tokenizador → preentrenamiento → representación aprendida → SFT → evaluación`. La idea de fondo es
*misma arquitectura, mejores datos → aparece el significado*: el mismo Transformer que sobre Shakespeare
escupía texto sin sentido, entrenado sobre cuentos infantiles, escribe cuentos coherentes.

Todo el código de arranque está acá:

**https://github.com/solidgoldmagickarp/proyecto_de_lenguaje_emergente**

No hace falta que se creen una cuenta de GitHub ni que clonen nada a mano. El notebook se encarga.

---

## 1 · Abran el notebook en Colab

Este es el camino principal y el que recomendamos. Hagan clic acá:

**https://colab.research.google.com/github/solidgoldmagickarp/proyecto_de_lenguaje_emergente/blob/master/00_dataset.ipynb**

Una vez adentro, **el orden importa**:

1. **Primero elijan la GPU:** *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4*.
   Cambiar el tipo de entorno reemplaza la máquina virtual y borra todo, así que si lo hacen después
   tiran a la basura lo que ya habían corrido.
2. **Después, `Archivo → Guardar una copia en Drive`.** El notebook que abrieron es de solo lectura: si
   no hacen la copia, pierden todo lo que escriban.
3. **Recién ahí corran la primera celda.** Esa celda clona el repositorio, entra en la carpeta e instala
   las dependencias. Todos los `import` que vienen después dependen de eso.

> Si en algún momento ven `ModuleNotFoundError: No module named 'environment'`, es que se recicló el
> entorno de ejecución. Vuelvan a correr la primera celda y sigan.

`00_dataset.ipynb` es la **Etapa 0**: bajar los datos, mirarlos y entender por qué este dataset es un
instrumento de laboratorio. Es lo único que viene hecho. **Las Etapas 1 a 5 las escriben ustedes**, en
los notebooks que vayan creando.

---

## 2 · Lean la consigna completa

El instructivo que están leyendo es solo la puerta de entrada. La consigna de verdad —las cinco etapas
paso a paso, los fragmentos de código adaptables, las trampas que se van a comer y las preguntas
conceptuales que arman la nota— está en el archivo **`assignment.md`** del repositorio:

**https://github.com/solidgoldmagickarp/proyecto_de_lenguaje_emergente/blob/master/assignment.md**

Léanlo entero **antes** de escribir código.

---

## 3 · Qué hay que entregar

Suban al campus **un solo archivo .zip** llamado `grupo_<>_<apellido>.zip` (elijan un apellido de cualquier integrante), con:

- **Los notebooks** (`.ipynb`) de las etapas que hayan hecho. **Con las salidas adentro**: si nos llega
  un notebook sin ejecutar, no vemos ni una curva ni una muestra generada. Antes de descargarlo,
  chequeen que se vean los resultados.
- **El informe**, en PDF o como markdown dentro de un notebook. Va: las decisiones de diseño y por qué
  las tomaron, curvas de pérdida y perplejidad, muestras generadas, la tabla de la ablación
  ancho/profundidad, el análisis de embeddings, la comparación base contra SFT, y las respuestas a las
  preguntas de cada etapa.
- **El registro de decisiones**: con qué contaban al arrancar, qué probaron, qué recortaron y por qué.
  Puede ir dentro del informe.

**No metan la carpeta `checkpoints/` en el zip.** Los `.pt` pesan y no los necesitamos.

---

## 4 · Ojo con esto: el disco de Colab se borra

Cuando se desconecta el entorno de ejecución (más o menos 90 minutos de inactividad, 12 horas como
máximo en la versión gratuita) **desaparece todo lo que hayan guardado en `checkpoints/`**, y las
Etapas 3, 4 y 5 dependen de los checkpoints que produce la Etapa 2.

Este proyecto es de tres semanas: no lo van a hacer de una sentada. Tienen dos salidas:

- **Montar Drive** y dejar `checkpoints/` ahí. La primera celda de cada notebook trae un bloque
  comentado que hace exactamente eso: descoméntenlo.
- **Bajar los archivos de checkpoint a mano** antes de cerrar la pestaña.

Perder un modelo entrenado por una desconexión es la forma más común en que un grupo pierde un día
entero de trabajo. Resuélvanlo el primer día, no el último.

---

## 5 · Si algo se rompe

Algo se va a romper: un dataset que no baja, una sesión que se cae en la mitad de una corrida, un
`CUDA out of memory`, un resultado que no reproduce. **Eso no es un contratiempo del proyecto: es el
proyecto.** No hay un camino correcto escondido que tengan que adivinar.

Si una etapa no corre como está planteada, encuéntrenle la vuelta: achiquen el subconjunto, cambien el
experimento por otro que conteste la misma pregunta, o cambien la pregunta por una que sí puedan
contestar con lo que les corre. Después cuenten en el registro qué se rompió, qué probaron y por qué
terminaron donde terminaron. Un desvío bien explicado es una entrega; un casillero vacío no.
