# %% [markdown]
r"""
# Etapa 5 · Medir la emergencia con un juez local

Qwen3 4B vía Ollama pone nota (1–10) a gramática, creatividad, consistencia y, en las salidas con instrucción,
obediencia. Replica la forma de evaluar del paper de TinyStories.

Tres cosas para que la nota del juez signifique algo:

- **Calibración:** además de los modelos, el juez corrige el modelo sin entrenar (piso) y cuentos reales del
  dataset (techo). Si no los separa bien, el resto de las notas no dice mucho.
- **A ciegas y mezclado:** el juez no ve qué modelo escribió cada texto, y los ítems van en orden aleatorio.
- **Ruido del propio juez:** un subconjunto se corrige dos veces con temperatura > 0.

Pensado para Colab con GPU. Localmente funciona si Ollama está instalado y corriendo.
"""

# %%
import environment  # noqa: F401

environment.summary()

SMOKE_TEST = False
SEED = 1337
DEVICE = environment.device()
JUDGE_MODEL = "qwen3:4b"
N_PER_MODEL = 3 if SMOKE_TEST else 25      # generaciones por modelo y por tarea
N_REPEAT = 3 if SMOKE_TEST else 20         # ítems corregidos dos veces para medir el ruido del juez

# %% [markdown]
r"""
## Levantar Ollama
"""

# %%
import shutil
import subprocess
import sys
import time

import requests

if shutil.which("ollama") is None and "google.colab" in sys.modules:
    subprocess.run("apt-get -qq install -y zstd && curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)


def ollama_up():
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


if not ollama_up():
    if shutil.which("ollama") is None:
        raise RuntimeError("No hay Ollama instalado. En Colab esta celda lo instala; localmente: https://ollama.com")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        if ollama_up():
            break
        time.sleep(1)
print("Ollama arriba:", ollama_up())
subprocess.run(["ollama", "pull", JUDGE_MODEL], check=True)

# %% [markdown]
r"""
## Generar

- **Tarea A, continuación:** las primeras palabras de un cuento de validación; todos los modelos.
- **Tarea B, instrucción:** un encabezado `Words:`/`Features:`/... de TinyStories-Instruct validación; el base
  ancho (para ver qué hace sin SFT) y los modelos con SFT.
"""

# %%
import json
import re
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from data import load_instruct_records, load_tinystories
from lab import generate_text, load_checkpoint

tokenizer = Tokenizer.from_file("checkpoints/tokenizer.json")

MODELS = {
    "step0 (piso)": "checkpoints/pretrain_step0.pt",
    "wide": "checkpoints/pretrain_wide.pt",
    "deep": "checkpoints/pretrain_deep.pt",
    "sft full": "checkpoints/sft_final.pt",
    "sft lora": "checkpoints/sft_lora.pt",
}
MODELS = {k: v for k, v in MODELS.items() if Path(v).exists()}
SFT_NAMES = [k for k in MODELS if k.startswith("sft")]
print("modelos:", list(MODELS))

val_stories = load_tinystories("validation", limit=200 + N_PER_MODEL)["text"][200:]  # distintos de los de otras etapas
openings = [" ".join(s.split()[:8]) for s in val_stories[:N_PER_MODEL]]

instruct = load_instruct_records("validation", limit=500)
# Mismo cuidado que en la Etapa 4: "Story:" puede traer el cuento en la misma línea, y algún campo puede venir
# después del cuento. El prompt usa siempre la forma "Story: \n\n", la mayoritaria en los datos.
FIELD_LINE = re.compile(r"^(Features|Words|Summary|Random sentence):[ \t]*.*$", re.MULTILINE)
headers = []
for rec in instruct:
    m = re.match(r"^(.*?)Story:[ \t]*\n*(.*)$", rec, re.DOTALL)
    if m and "Words:" in m.group(1):
        trailing = FIELD_LINE.search(m.group(2))
        story = m.group(2)[: trailing.start()].rstrip() if trailing else m.group(2)
        headers.append((m.group(1).rstrip("\n") + "\nStory: \n\n", story))
    if len(headers) >= N_PER_MODEL:
        break

items = []
for name, path in MODELS.items():
    model, _ = load_checkpoint(path, DEVICE)
    for i, opening in enumerate(openings):
        items.append({"model": name, "task": "continuation", "prompt": opening,
                      "completion": generate_text(model, tokenizer, opening, max_new_tokens=180, seed=SEED + i)})
    if name in SFT_NAMES or name == "wide":
        for i, (header, _) in enumerate(headers):
            items.append({"model": name, "task": "instruction", "prompt": header,
                          "completion": generate_text(model, tokenizer, header, max_new_tokens=200, seed=SEED + i)})
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# techo: cuentos reales (continuación del mismo comienzo, y cuento real para el mismo encabezado)
for opening, story in zip(openings, val_stories):
    items.append({"model": "dataset (techo)", "task": "continuation", "prompt": opening,
                  "completion": story[len(opening):]})
for header, story in headers:
    items.append({"model": "dataset (techo)", "task": "instruction", "prompt": header, "completion": story})

print(f"ítems a corregir: {len(items)}")

# %% [markdown]
r"""
## El juez

El formato sigue al paper: se muestra el comienzo y la parte escrita por el alumno separada con `***`, y se pide
corregir como un docente. `format="json"` obliga a JSON válido; igual se parsea con `raw_decode` (corta al final
del primer objeto) y cada llamada está envuelta para que una respuesta mala no tire abajo la corrida.
"""

# %%
import ollama

CRITERIA = ["grammar", "creativity", "consistency"]

PROMPT_CONTINUATION = """The following exercise asks a student to complete a short story for young children. \
The beginning of the story was given to the student; the student's completion is the part after the *** marks.

{prompt}***{completion}

Grade the student's completion as a strict but fair teacher would. The story may be cut off at the end because of \
a length limit; do not penalize that. Give each criterion an integer from 1 (very poor) to 10 (excellent):
- grammar: correct English grammar and spelling
- creativity: interesting, non-generic content
- consistency: the completion is coherent with the beginning and with itself (characters, objects, events)

Answer ONLY with a JSON object: {{"grammar": int, "creativity": int, "consistency": int, "comment": "one sentence"}}"""

PROMPT_INSTRUCTION = """The following exercise asks a student to write a short story for young children that \
satisfies the instructions given before "Story:". The student's story is the part after the *** marks.

{prompt}***{completion}

Grade the student's story as a strict but fair teacher would. The story may be cut off at the end because of a \
length limit; do not penalize that. Give each criterion an integer from 1 (very poor) to 10 (excellent):
- grammar: correct English grammar and spelling
- creativity: interesting, non-generic content
- consistency: the story is coherent with itself (characters, objects, events)
- instruction_following: the story uses the required words and satisfies the features, summary or sentence given

Answer ONLY with a JSON object: \
{{"grammar": int, "creativity": int, "consistency": int, "instruction_following": int, "comment": "one sentence"}}"""


def parse_grade(text, criteria):
    start = text.find("{")
    if start < 0:
        raise ValueError("no hay JSON en la respuesta")
    obj, _ = json.JSONDecoder().raw_decode(text, start)
    grade = {}
    for c in criteria:
        v = obj[c]
        v = int(round(float(v)))
        if not 1 <= v <= 10:
            raise ValueError(f"{c}={v} fuera de rango")
        grade[c] = v
    grade["comment"] = str(obj.get("comment", ""))
    return grade


def judge(item, temperature=0.0):
    criteria = CRITERIA + (["instruction_following"] if item["task"] == "instruction" else [])
    template = PROMPT_INSTRUCTION if item["task"] == "instruction" else PROMPT_CONTINUATION
    content = template.format(prompt=item["prompt"].rstrip(), completion=item["completion"])
    raw = ""
    try:
        response = ollama.chat(model=JUDGE_MODEL, messages=[{"role": "user", "content": content}],
                               format="json", think=False, options={"temperature": temperature, "seed": SEED})
        raw = response["message"]["content"]
        return {"ok": True, **parse_grade(raw, criteria)}
    except Exception as exc:  # respuesta mal formada, timeout, etc.: se registra y se sigue
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "raw": raw[:300]}


rng = np.random.default_rng(SEED)
order = rng.permutation(len(items))
t0 = time.time()
for n, k in enumerate(order, start=1):
    items[k]["grade"] = judge(items[k])
    if n % 10 == 0 or n == len(items):
        print(f"{n}/{len(items)} corregidos ({time.time() - t0:.0f}s)")

failures = [it for it in items if not it["grade"]["ok"]]
print(f"respuestas del juez que no se pudieron usar: {len(failures)}/{len(items)}")
for it in failures[:3]:
    print("  ", it["grade"]["error"], "|", repr(it["grade"]["raw"]))

# %% [markdown]
r"""
## Ruido del juez

El mismo ítem, corregido dos veces con temperatura 0.8. Si la diferencia típica entre dos lecturas es del mismo
tamaño que la diferencia entre dos modelos, el juez no los está separando.
"""

# %%
repeat_idx = rng.choice([k for k, it in enumerate(items) if it["grade"]["ok"]], size=min(N_REPEAT, len(items)), replace=False)
diffs = {c: [] for c in CRITERIA}
for k in repeat_idx:
    a, b = judge(items[k], temperature=0.8), judge(items[k], temperature=0.8)
    if a["ok"] and b["ok"]:
        for c in CRITERIA:
            diffs[c].append(abs(a[c] - b[c]))
for c, d in diffs.items():
    print(f"{c:<12}: |diferencia| media entre dos lecturas = {np.mean(d):.2f} (n={len(d)})")

# %% [markdown]
r"""
## Resultados
"""

# %%
import matplotlib.pyplot as plt

ALL_CRITERIA = CRITERIA + ["instruction_following"]
model_order = list(MODELS) + ["dataset (techo)"]
table = {}
for task in ("continuation", "instruction"):
    for name in model_order:
        graded = [it["grade"] for it in items if it["model"] == name and it["task"] == task and it["grade"]["ok"]]
        if not graded:
            continue
        row = {}
        for c in ALL_CRITERIA:
            vals = [g[c] for g in graded if c in g]
            if vals:
                row[c] = (float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals))), len(vals))
        table[(task, name)] = row

for task in ("continuation", "instruction"):
    print(f"\n=== {task} === (media ± error estándar)")
    print(f"{'modelo':<17}" + "".join(f"{c[:13]:>18}" for c in ALL_CRITERIA))
    for name in model_order:
        row = table.get((task, name))
        if row:
            print(f"{name:<17}" + "".join(f"{row[c][0]:>11.2f} ±{row[c][1]:.2f}" if c in row else f"{'-':>18}"
                                          for c in ALL_CRITERIA))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, task in zip(axes, ("continuation", "instruction")):
    names = [n for n in model_order if (task, n) in table]
    crits = [c for c in ALL_CRITERIA if any(c in table[(task, n)] for n in names)]
    width = 0.8 / max(1, len(names))
    for j, n in enumerate(names):
        means = [table[(task, n)].get(c, (np.nan, 0, 0))[0] for c in crits]
        errs = [table[(task, n)].get(c, (0, 0, 0))[1] for c in crits]
        ax.bar(np.arange(len(crits)) + j * width, means, width, yerr=errs, label=n, capsize=2)
    ax.set_xticks(np.arange(len(crits)) + width * (len(names) - 1) / 2, crits, fontsize=8)
    ax.set(ylim=(0, 10), title=f"juez · {task}")
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("checkpoints/stage5_judge.png", dpi=110)
plt.show()

# %% [markdown]
r"""
## Juez contra perplejidad

Perplejidad de validación de la Etapa 2 (TinyStories plano) junto a la nota media del juez en la tarea de
continuación. Con tan pocos modelos no hay correlación que calcular: la pregunta es si el *orden* coincide.
"""

# %%
import math

ppl = {}
for name, path in MODELS.items():
    hist = Path(path).with_name(Path(path).stem + "_history.json")
    if hist.exists():
        h = json.loads(hist.read_text(encoding="utf-8"))
        h = h.get("history", h)
        ppl[name] = math.exp(h["val_loss"][-1])
print("(la perplejidad de los modelos SFT es sobre Instruct, no es comparable directamente con la de wide/deep)")
print(f"{'modelo':<17} {'ppl val':>9} {'nota media (continuación)':>26}")
for name in MODELS:
    row = table.get(("continuation", name))
    mean_score = np.mean([row[c][0] for c in CRITERIA]) if row else float("nan")
    print(f"{name:<17} {ppl.get(name, float('nan')):>9.2f} {mean_score:>26.2f}")

with open("checkpoints/stage5_judge.json", "w", encoding="utf-8") as f:
    json.dump({"items": items, "judge_noise": {c: float(np.mean(d)) if d else None for c, d in diffs.items()},
               "smoke_test": SMOKE_TEST}, f, ensure_ascii=False, indent=1)

# %% [markdown]
r"""
## Para el informe

- ¿El juez separa el piso del techo? Si no, ¿qué valen el resto de sus notas?
- ¿Dónde le da la razón a la perplejidad y dónde se va para otro lado? ¿Qué dice eso sobre lo que mide cada uno?
- ¿Las diferencias entre wide y deep, o entre full FT y LoRA, superan el ruido del propio juez?
- Lean algunos `comment` del juez: ¿corrige lo que dice que corrige?
"""
