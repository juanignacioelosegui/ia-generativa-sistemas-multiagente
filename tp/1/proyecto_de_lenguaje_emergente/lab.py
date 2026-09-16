"""
Piezas compartidas entre las Etapas 2 a 5: el Transformer, checkpoints, batches, entrenamiento y generación.

    from lab import GPTConfig, GPTLanguageModel, train, generate_text, save_checkpoint, load_checkpoint

POR QUÉ UN MÓDULO Y NO UNA CELDA POR NOTEBOOK
---------------------------------------------
La Etapa 3 reconstruye el modelo desde un checkpoint, la Etapa 4 lo sigue entrenando y la Etapa 5 genera con
él. Si cada notebook tuviera su propia copia de la clase, una diferencia mínima (un `bias=False` de más) haría
fallar `load_state_dict` recién en la etapa siguiente. Una sola definición evita eso.

OJO EN COLAB: la celda de arranque clona el repositorio de la cátedra, que NO tiene este archivo. O suben este
repositorio a uno propio público y cambian `REPO_URL` en `tools/py_to_notebook.py`, o suben `lab.py` a mano a la
carpeta del repositorio clonado antes de correr la segunda celda.

La arquitectura es la de siempre (`Head` -> `MultiHeadAttention` -> `FeedForward` -> `Block` -> `GPTLanguageModel`),
sin weight tying entre la tabla de embeddings y `lm_head` -- así la cuenta de parámetros de la §7 de la consigna
(ancho vs. profundo a <7%) se reproduce tal cual. Lo único distinto a la versión de clase es que `Head` llama a
`F.scaled_dot_product_attention` en vez de armar la matriz de atención a mano: es matemáticamente lo mismo
(máscara causal + softmax escalado + dropout), pero en GPU corre con un kernel fusionado y usa menos memoria.
`Head.key/query/value` y `MultiHeadAttention.proj` siguen siendo `nn.Linear` separados, que es lo que necesita
la LoRA hecha a mano de la Etapa 4.
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

EOS_TOKEN = "<|endoftext|>"


# =============================================================================
# modelo
# =============================================================================
@dataclass
class GPTConfig:
    vocab_size: int = 8192
    n_embd: int = 256
    n_layer: int = 2
    n_head: int = 8
    block_size: int = 192
    dropout: float = 0.1


class Head(nn.Module):
    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = dropout

    def forward(self, x):
        k, q, v = self.key(x), self.query(x), self.value(x)
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd={n_embd} no es divisible por n_head={n_head}"
        self.sa = MultiHeadAttention(n_embd, n_head, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config.n_embd, config.n_head, config.dropout) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding_table(idx)
        pos = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(tok + pos))
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


def count_params(model: nn.Module) -> dict[str, int]:
    """Desglose de parámetros: dónde está el grueso del modelo importa para emparejar la ablación."""
    tok = model.token_embedding_table.weight.numel()
    pos = model.position_embedding_table.weight.numel()
    head = sum(p.numel() for p in model.lm_head.parameters())
    blocks = sum(p.numel() for p in model.blocks.parameters())
    total = sum(p.numel() for p in model.parameters())
    return {
        "token_embedding": tok,
        "position_embedding": pos,
        "blocks": blocks,
        "ln_f": total - tok - pos - head - blocks,
        "lm_head": head,
        "total": total,
    }


# =============================================================================
# checkpoints
# =============================================================================
def save_checkpoint(path, model: GPTLanguageModel, **extra) -> None:
    """Guarda `state_dict` + la config que define la forma: sin eso el checkpoint no se puede reconstruir."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict(), **extra}, path)
    print(f"checkpoint guardado: {path} ({path.stat().st_size / 1e6:.1f} MB)")


def load_checkpoint(path, device="cpu") -> tuple[GPTLanguageModel, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPTLanguageModel(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


# =============================================================================
# datos
# =============================================================================
def encode_texts(tokenizer, texts, eos_id: int, batch_size: int = 10_000) -> np.ndarray:
    """
    Tokeniza una lista de textos y los concatena en un único arreglo 1D, con `<|endoftext|>` después de cada
    uno. Así el modelo ve fronteras entre cuentos y aprende a cerrarlos. `uint16` alcanza para vocab <= 65.535
    y ocupa la mitad que `int32`: con cientos de millones de tokens eso es la diferencia entre entrar en la
    RAM de Colab o no.
    """
    assert tokenizer.get_vocab_size() <= 65_535
    chunks = []
    for start in range(0, len(texts), batch_size):
        for enc in tokenizer.encode_batch(list(texts[start : start + batch_size])):
            chunks.append(np.asarray(enc.ids + [eos_id], dtype=np.uint16))
    return np.concatenate(chunks)


def get_batch(data: np.ndarray, block_size: int, batch_size: int, device, rng: np.random.Generator):
    """Ventanas al azar de largo `block_size`; el target es la misma ventana corrida un token."""
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix]).astype(np.int64)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    if str(device).startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


# =============================================================================
# entrenamiento
# =============================================================================
def _autocast(device):
    # fp16 y no bf16: la T4 de Colab no soporta bf16 de forma nativa.
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def estimate_loss(model, splits: dict, eval_iters, batch_size, device, seed=0, batch_fn=None):
    """Pérdida media sobre `eval_iters` batches por split. Semilla fija: todas las evaluaciones ven los mismos batches."""
    batch_fn = batch_fn or get_batch
    model.eval()
    out = {}
    for name, data in splits.items():
        rng = np.random.default_rng(seed)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = batch_fn(data, model.config.block_size, batch_size, device, rng)
            with _autocast(device):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def train(
    model,
    train_data: np.ndarray,
    val_data: np.ndarray,
    *,
    device,
    max_steps: int,
    batch_size: int = 64,
    lr: float = 3e-4,
    min_lr_ratio: float = 0.1,
    warmup_steps: int = 100,
    weight_decay: float = 0.1,
    eval_interval: int = 250,
    eval_iters: int = 50,
    sample_steps: tuple[int, ...] = (),
    sample_fn=None,
    seed: int = 1337,
    params=None,
    batch_fn=None,
):
    """
    El ciclo `forward -> loss -> backward -> update` de siempre, con tres agregados de ingeniería:
    AdamW con warmup + decaimiento coseno, mixed precision en GPU y clipping de gradiente en 1.0.

    `sample_fn(step)` se llama en cada paso de `sample_steps` (antes del update de ese paso), para leer muestras
    a lo largo del entrenamiento. `params` permite entrenar solo un subconjunto (LoRA). `batch_fn` reemplaza a
    `get_batch` con la misma firma (el SFT arma batches alineados al comienzo de cada ejemplo). Devuelve el historial.
    """
    batch_fn = batch_fn or get_batch
    model.train()
    params = [p for p in (params if params is not None else model.parameters()) if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))

    def lr_at(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))

    rng = np.random.default_rng(seed)
    history = {"step": [], "train_loss": [], "val_loss": [], "lr": [], "tokens": [], "elapsed_s": []}
    splits = {"train": train_data, "val": val_data}
    block = model.config.block_size
    sample_steps = set(sample_steps)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    for step in range(max_steps + 1):
        if step % eval_interval == 0 or step == max_steps:
            losses = estimate_loss(model, splits, eval_iters, batch_size, device, batch_fn=batch_fn)
            elapsed = time.time() - t0
            for key, value in (
                ("step", step), ("train_loss", losses["train"]), ("val_loss", losses["val"]),
                ("lr", lr_at(step)), ("tokens", step * batch_size * block), ("elapsed_s", elapsed),
            ):
                history[key].append(value)
            print(
                f"paso {step:>6} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| ppl val {math.exp(losses['val']):8.2f} | lr {lr_at(step):.2e} | {elapsed:6.0f}s"
            )
        if sample_fn is not None and step in sample_steps:
            model.eval()
            sample_fn(step)
            model.train()
        if step == max_steps:
            break

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        x, y = batch_fn(train_data, block, batch_size, device, rng)
        with _autocast(device):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()

    history["peak_mem_mb"] = (
        torch.cuda.max_memory_allocated() / 1e6 if str(device).startswith("cuda") else float("nan")
    )
    model.eval()
    return history


# =============================================================================
# generación
# =============================================================================
@torch.no_grad()
def generate(model, idx, max_new_tokens, *, temperature=1.0, top_k=None, greedy=False, eos_id=None, generator=None):
    model.eval()
    block = model.config.block_size
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block:])
        logits = logits[:, -1, :].float()
        if greedy:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
        idx = torch.cat([idx, next_id], dim=1)
        if eos_id is not None and (next_id == eos_id).all():
            break
    return idx


def generate_text(
    model, tokenizer, prompt: str, *, max_new_tokens=150, seed=1337, temperature=0.8, top_k=40,
    greedy=False, stop_at_eos=True,
) -> str:
    """
    Genera una continuación de `prompt` y devuelve SOLO la parte nueva.

    La semilla va en un `torch.Generator` propio, no en `torch.manual_seed` global: el que llama decide la
    semilla por prompt (`seed + i`, ver la trampa de la Etapa 2) sin pisar el estado aleatorio de nadie más.
    """
    device = next(model.parameters()).device
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    ids = tokenizer.encode(prompt).ids if prompt else [eos_id]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    out = generate(
        model, idx, max_new_tokens, temperature=temperature, top_k=top_k, greedy=greedy,
        eos_id=eos_id if stop_at_eos else None, generator=gen,
    )
    new_ids = [t for t in out[0, len(ids):].tolist() if t != eos_id]
    return tokenizer.decode(new_ids)
