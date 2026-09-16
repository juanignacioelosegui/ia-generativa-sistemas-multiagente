# %% [markdown]
r"""
# Etapa 4 · SFT — comportamiento, no conocimiento

Seguimos entrenando el modelo ancho de la Etapa 2 sobre TinyStories-Instruct, con la misma pérdida de próximo
token. Lo único que cambia son los datos: bloques `encabezado de instrucción → Story: → cuento`.

Dos decisiones de implementación que conviene tener presentes al leer los resultados:

- **Batches alineados al ejemplo.** En el preentrenamiento las ventanas caen en cualquier lugar del corpus. Acá
  cada fila del batch arranca en el comienzo de un registro, así el modelo siempre ve la instrucción entera antes
  del cuento. Lo que no entra en `block_size` se trunca; lo que sobra se rellena y queda fuera de la pérdida.
- **La pérdida incluye el encabezado.** Es el formato de continuación pelado que describe la consigna; no se
  enmascaran los tokens de la instrucción.

Escribe `checkpoints/sft_final.pt` (fine-tuning completo) y, si `DO_LORA`, `checkpoints/sft_lora.pt` (con el
adaptador ya fusionado en los pesos, para que se cargue como cualquier otro modelo).
"""

# %%
import environment  # noqa: F401

environment.summary()

SMOKE_TEST = False
RETRAIN = False
DO_LORA = True
SEED = 1337

DEVICE = environment.device()
SFT_RECORDS = 1_000 if SMOKE_TEST else 100_000
VAL_RECORDS = 200 if SMOKE_TEST else 2_000
BATCH_SIZE = 16 if SMOKE_TEST else 64
MAX_STEPS = 100 if SMOKE_TEST else 2_000
LR = 1e-4
LORA_RANK, LORA_ALPHA, LORA_LR = 8, 16, 1e-3
EVAL_INTERVAL = 25 if SMOKE_TEST else 200
EVAL_ITERS = 10 if SMOKE_TEST else 40
N_EVAL_PROMPTS = 10 if SMOKE_TEST else 100

# %% [markdown]
r"""
## Datos

`load_instruct_records` reagrupa el dataset de una-línea-por-fila en registros completos. La primera vez tarda:
materializa el split entero antes de cortar (ver `data.py`).
"""

# %%
import json
import re
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from data import load_instruct_records
from lab import EOS_TOKEN

tokenizer = Tokenizer.from_file("checkpoints/tokenizer.json")
EOS_ID = tokenizer.token_to_id(EOS_TOKEN)

train_records = load_instruct_records("train", limit=SFT_RECORDS)
val_records = load_instruct_records("validation", limit=VAL_RECORDS + N_EVAL_PROMPTS)
eval_records, val_records = val_records[:N_EVAL_PROMPTS], val_records[N_EVAL_PROMPTS:]
print(f"registros: train {len(train_records):,} | val {len(val_records):,} | prompts de evaluación {len(eval_records):,}")
print("\n--- un registro, tal cual entra al modelo ---")
print(repr(train_records[0][:400]))

# %%
# MEDIDO sobre el dataset (no asumido): conviven dos formatos. ~55% trae "Story: \n\n<cuento>" y ~45% trae
# "Story: <cuento>" en la misma línea; y en algunos registros un campo (típicamente Summary:) aparece DESPUÉS
# del cuento. Por eso: `\s*` después de "Story:", y los campos se buscan en cualquier línea del registro.
STORY_SPLIT = re.compile(r"^(.*?Story:[ \t]*\n*)(.*)$", re.DOTALL)
HEADER_KEYS = ("Features", "Words", "Summary", "Random sentence")
FIELD_LINE = re.compile(rf"^({'|'.join(HEADER_KEYS)}):[ \t]*(.*)$", re.MULTILINE)


def split_record(rec):
    """-> (encabezado hasta 'Story:' inclusive, cuento, dict de campos). None si el registro no tiene 'Story:'."""
    m = STORY_SPLIT.match(rec)
    if not m:
        return None
    header, body = m.group(1), m.group(2)
    trailing = FIELD_LINE.search(body)
    story = body[: trailing.start()].rstrip() if trailing else body
    fields = {k: v.strip() for k, v in FIELD_LINE.findall(header)}
    fields.update({k: v.strip() for k, v in FIELD_LINE.findall(body)})
    return header, story, fields


parsed = [split_record(r) for r in train_records]
print(f"registros sin 'Story:': {sum(p is None for p in parsed)}")
print(f"'Story:' con el cuento en la misma línea: {np.mean([not p[0].endswith(chr(10)) for p in parsed if p]):.0%}")
print(f"con algún campo DESPUÉS del cuento: {np.mean([bool(FIELD_LINE.search(r.split('Story:', 1)[1])) for r in train_records if 'Story:' in r]):.0%}")
field_freq = {k: np.mean([k in p[2] for p in parsed if p]) for k in HEADER_KEYS}
print("fracción de registros con cada campo:", {k: f"{v:.0%}" for k, v in field_freq.items()})
orders = {tuple(k for k in [ln.partition(':')[0] for ln in p[0].splitlines()] if k in HEADER_KEYS)
          for p in parsed[:2000] if p}
print(f"órdenes distintos de campos en los primeros 2.000 registros: {len(orders)}")

# %%
from lab import load_checkpoint

base_model, base_ckpt = load_checkpoint("checkpoints/pretrain_wide.pt", DEVICE)
BLOCK = base_model.config.block_size


def encode_records(records):
    """Matriz (N, BLOCK+1) alineada al comienzo de cada registro, rellenada con EOS, y el largo real de cada uno."""
    seqs = np.full((len(records), BLOCK + 1), EOS_ID, dtype=np.int32)
    lengths = np.zeros(len(records), dtype=np.int32)
    for k, enc in enumerate(tokenizer.encode_batch(records)):
        ids = (enc.ids + [EOS_ID])[: BLOCK + 1]
        seqs[k, : len(ids)] = ids
        lengths[k] = len(ids)
    return {"seqs": seqs, "lengths": lengths}


def record_batch(data, block_size, batch_size, device, rng):
    ix = rng.integers(0, len(data["seqs"]), size=batch_size)
    seq = torch.from_numpy(data["seqs"][ix].astype(np.int64))
    x, y = seq[:, :-1].clone(), seq[:, 1:].clone()
    # posiciones de relleno: fuera de la pérdida (-100 es el ignore_index por defecto de cross_entropy)
    pad = torch.arange(block_size)[None, :] >= torch.from_numpy(data["lengths"][ix] - 1)[:, None]
    y[pad] = -100
    return x.to(device), y.to(device)


train_data = encode_records(train_records)
val_data = encode_records(val_records)
full_lengths = np.array([len(e.ids) + 1 for e in tokenizer.encode_batch(train_records[:5000])])
print(f"tokens por registro: media {full_lengths.mean():.0f}, p90 {np.percentile(full_lengths, 90):.0f}; "
      f"entran enteros en {BLOCK}: {(full_lengths <= BLOCK + 1).mean():.0%}")

# %% [markdown]
r"""
## Fine-tuning completo
"""

# %%
import copy

from lab import count_params, save_checkpoint, train

# pérdida del modelo base sobre el Instruct, ANTES de tocar nada: el punto de partida de la curva
base_instruct = __import__("lab").estimate_loss(base_model, {"val": val_data}, EVAL_ITERS, BATCH_SIZE, DEVICE,
                                                batch_fn=record_batch)["val"]
print(f"modelo base, pérdida sobre Instruct val: {base_instruct:.3f}")


def train_or_load(path, build_and_train):
    path = Path(path)
    hist_path = path.with_name(path.stem + "_history.json")
    if path.exists() and hist_path.exists() and not RETRAIN:
        model, _ = load_checkpoint(path, DEVICE)
        print(f"cargado de {path}")
        return model, json.loads(hist_path.read_text(encoding="utf-8"))
    model, history = build_and_train()
    save_checkpoint(path, model, base="pretrain_wide.pt", steps=MAX_STEPS, smoke_test=SMOKE_TEST)
    hist_path.write_text(json.dumps(history, indent=1), encoding="utf-8")
    return model, history


def full_ft():
    torch.manual_seed(SEED)
    model = copy.deepcopy(base_model)
    history = train(model, train_data, val_data, device=DEVICE, max_steps=MAX_STEPS, batch_size=BATCH_SIZE, lr=LR,
                    warmup_steps=min(50, MAX_STEPS // 10), eval_interval=EVAL_INTERVAL, eval_iters=EVAL_ITERS,
                    seed=SEED, batch_fn=record_batch)
    history["trainable_params"] = sum(p.numel() for p in model.parameters())
    return model, history


sft_model, sft_history = train_or_load("checkpoints/sft_final.pt", full_ft)

# %% [markdown]
r"""
## LoRA hecha a mano (opcional)

`LoRALayer` y `LinearWithLoRA` son las de la consigna. Se envuelven `Head.key/query/value` y
`MultiHeadAttention.proj` de cada bloque; todo lo demás queda congelado. Mismos datos, mismos pasos, `lr` más alto
(es habitual en LoRA: se entrenan pocos parámetros que arrancan en cero).
"""

# %%
import math

import torch.nn as nn


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


class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha).to(
            device=linear.weight.device, dtype=linear.weight.dtype
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

    def merged(self):
        """Un nn.Linear común con ΔW sumado: W tiene forma (out, in) y x @ A @ B equivale a x @ (A @ B), de ahí el .T"""
        out = copy.deepcopy(self.linear)
        with torch.no_grad():
            out.weight += (self.lora.scaling * (self.lora.A @ self.lora.B)).T.to(out.weight.dtype)
        for p in out.parameters():
            p.requires_grad = True
        return out


def add_lora(model, rank, alpha):
    for p in model.parameters():
        p.requires_grad = False
    for block in model.blocks:
        for head in block.sa.heads:
            head.key = LinearWithLoRA(head.key, rank, alpha)
            head.query = LinearWithLoRA(head.query, rank, alpha)
            head.value = LinearWithLoRA(head.value, rank, alpha)
        block.sa.proj = LinearWithLoRA(block.sa.proj, rank, alpha)
    return model


def merge_lora(model):
    for block in model.blocks:
        for head in block.sa.heads:
            head.key, head.query, head.value = head.key.merged(), head.query.merged(), head.value.merged()
        block.sa.proj = block.sa.proj.merged()
    return model


def lora_ft():
    torch.manual_seed(SEED)
    model = add_lora(copy.deepcopy(base_model), LORA_RANK, LORA_ALPHA)
    trainable = [p for p in model.parameters() if p.requires_grad]
    history = train(model, train_data, val_data, device=DEVICE, max_steps=MAX_STEPS, batch_size=BATCH_SIZE,
                    lr=LORA_LR, warmup_steps=min(50, MAX_STEPS // 10), eval_interval=EVAL_INTERVAL,
                    eval_iters=EVAL_ITERS, seed=SEED, params=trainable, batch_fn=record_batch)
    history["trainable_params"] = sum(p.numel() for p in trainable)
    merged = merge_lora(model)
    # chequeo: fusionar no puede cambiar la salida
    return merged, history


if DO_LORA:
    lora_model, lora_history = train_or_load("checkpoints/sft_lora.pt", lora_ft)

# %%
import matplotlib.pyplot as plt

total_params = count_params(base_model)["total"]
runs = {"full FT": sft_history, **({f"LoRA r={LORA_RANK}": lora_history} if DO_LORA else {})}
print(f"{'corrida':<12} {'params entrenados':>18} {'fracción':>9} {'val final':>9} {'pico mem MB':>11} {'minutos':>8}")
for name, h in runs.items():
    print(f"{name:<12} {h['trainable_params']:>18,} {h['trainable_params'] / total_params:>9.2%} "
          f"{h['val_loss'][-1]:>9.3f} {h['peak_mem_mb']:>11.0f} {h['elapsed_s'][-1] / 60:>8.1f}")

fig, ax = plt.subplots(figsize=(6, 3.5))
for name, h in runs.items():
    ax.plot(h["step"], h["val_loss"], label=f"{name} val")
    ax.plot(h["step"], h["train_loss"], ":", label=f"{name} train")
ax.axhline(base_instruct, color="gray", ls="--", lw=1, label="base (sin SFT)")
ax.set(xlabel="paso de SFT", ylabel="pérdida sobre Instruct", title="SFT")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("checkpoints/stage4_sft_loss.png", dpi=110)
plt.show()

# %% [markdown]
r"""
## ¿Se olvidó de escribir cuentos sin instrucción?

Pérdida sobre TinyStories *plano* (el val de la Etapa 2), antes y después del SFT.
"""

# %%
from lab import estimate_loss

plain_val = np.load(f"checkpoints/tokens_{'smoke' if SMOKE_TEST else 'full'}.npz")["val"]
models_eval = {"base": base_model, "full FT": sft_model, **({"LoRA": lora_model} if DO_LORA else {})}
for name, m in models_eval.items():
    plain = estimate_loss(m, {"v": plain_val}, EVAL_ITERS, BATCH_SIZE, DEVICE)["v"]
    instr = estimate_loss(m, {"v": val_data}, EVAL_ITERS, BATCH_SIZE, DEVICE, batch_fn=record_batch)["v"]
    print(f"{name:<8} | TinyStories plano: {plain:.3f} | Instruct: {instr:.3f}")

# %% [markdown]
r"""
## El experimento que importa: mismo prompt, base contra SFT

Primero el prompt de la consigna, después prompts armados con los encabezados de registros de validación.
"""

# %%
from lab import generate_text


def make_prompt(fields):
    return "\n".join(f"{k}: {v}" for k, v in fields.items()) + "\nStory: \n\n"


# el prompt tiene que tener exactamente la forma de los datos: lo verificamos contra un registro real
example_header = split_record(train_records[0])[0]
assert make_prompt(split_record(train_records[0])[2]) == example_header, (
    f"el formato del prompt no coincide con los datos:\n{make_prompt(split_record(train_records[0])[2])!r}\n"
    f"vs\n{example_header!r}"
)

HANDMADE = [
    {"Words": "dragon, happy, forest"},
    {"Words": "boat, rain, sister"},
    {"Features": "Dialogue, BadEnding", "Words": "cookie, jump, blue"},
    {"Summary": "A little frog learns to share his toys with his friends."},
]
for i, fields in enumerate(HANDMADE):
    prompt = make_prompt(fields)
    print("=" * 90)
    print(prompt.strip())
    for name, m in models_eval.items():
        print(f"\n--- {name} ---")
        print(generate_text(m, tokenizer, prompt, max_new_tokens=180, seed=SEED + i))

# %% [markdown]
r"""
## Medirlo en vez de leer cuatro ejemplos

Sobre `N_EVAL_PROMPTS` registros de validación que tienen campo `Words:`, generamos con cada modelo (misma
semilla por prompt) y medimos:

- **contenido**: fracción de las palabras pedidas que aparecen en el cuento (se aceptan formas `-s/-es/-ed/-ing`).
  El *control* es la misma medida contra las palabras de *otro* prompt: lo que da el azar de un vocabulario chico.
- **forma**: si la salida empieza como un cuento (y no repitiendo líneas `Words:`/`Summary:`), y si cierra con
  `<|endoftext|>` antes del límite.
"""

# %%
def word_hit(word, text):
    w = re.escape(word.lower())
    return re.search(rf"\b{w}(s|es|ed|d|ing)?\b", text.lower()) is not None


def raw_generation(model, prompt, seed):
    """Como generate_text pero sin cortar en EOS al decodificar, para saber si cerró el cuento."""
    ids = tokenizer.encode(prompt).ids
    from lab import generate
    x = torch.tensor([ids], device=DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    out = generate(model, x, 200, temperature=0.8, top_k=40, eos_id=EOS_ID, generator=g)[0, len(ids):].tolist()
    closed = EOS_ID in out
    return tokenizer.decode([t for t in out if t != EOS_ID]), closed


eval_items = []
for rec in eval_records:
    p = split_record(rec)
    if p and "Words" in p[2]:
        eval_items.append({"fields": p[2], "words": [w.strip() for w in p[2]["Words"].split(",") if w.strip()]})
print(f"prompts de evaluación con Words: {len(eval_items)}")

instruction_results = {}
generations = {}
for name, m in models_eval.items():
    hits, control, story_form, closed_frac = [], [], [], []
    generations[name] = []
    for i, item in enumerate(eval_items):
        text, closed = raw_generation(m, make_prompt(item["fields"]), SEED + i)
        other = eval_items[(i + 1) % len(eval_items)]["words"]
        hits.append(np.mean([word_hit(w, text) for w in item["words"]]))
        control.append(np.mean([word_hit(w, text) for w in other]))
        story_form.append(not re.match(r"\s*(Features|Words|Summary|Random sentence|Story)\s*:", text))
        closed_frac.append(closed)
        generations[name].append({"prompt": make_prompt(item["fields"]), "text": text})
    instruction_results[name] = {
        "word_coverage": float(np.mean(hits)), "control_coverage": float(np.mean(control)),
        "starts_as_story": float(np.mean(story_form)), "closed_with_eos": float(np.mean(closed_frac)),
        "coverage_stderr": float(np.std(hits) / np.sqrt(len(hits))),
    }

print(f"{'modelo':<8} {'palabras pedidas':>17} {'control (azar)':>15} {'empieza cuento':>15} {'cierra con EOS':>15}")
for name, r in instruction_results.items():
    print(f"{name:<8} {r['word_coverage']:>11.1%} ±{r['coverage_stderr']:.1%} {r['control_coverage']:>15.1%} "
          f"{r['starts_as_story']:>15.0%} {r['closed_with_eos']:>15.0%}")

with open("checkpoints/stage4_sft.json", "w", encoding="utf-8") as f:
    json.dump({"instruction_results": instruction_results, "generations": generations,
               "base_instruct_loss": base_instruct, "smoke_test": SMOKE_TEST}, f, ensure_ascii=False, indent=1)

# %% [markdown]
r"""
## Para el informe

- Muestren el par antes/después con el mismo prompt: ¿qué cambió, la forma o el contenido? ¿Coincide lo que ven
  leyendo con lo que dicen las métricas de arriba?
- ¿Cuánto de la cobertura de palabras del modelo SFT le gana al control? ¿Y el base?
- Si hicieron LoRA: ¿qué fracción de los parámetros entrenaron, y cuánto de la mejora del fine-tuning completo
  compró eso? ¿Cambió la memoria pico tanto como esperaban? (a esta escala, piensen qué parte de la memoria son
  las activaciones y qué parte los estados del optimizador).
"""
