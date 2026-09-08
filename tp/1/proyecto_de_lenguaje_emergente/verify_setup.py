"""
Chequea que tu entorno esté listo para la consigna.

    python verify_setup.py

Corré esto ANTES de empezar, e idealmente mientras todavía tengas internet: el chequeo 3
hace una llamada chica de metadatos contra el HuggingFace Hub (no descarga
el dataset completo de TinyStories -- eso pasa una sola vez, dentro de 00_dataset, y solo
cuando efectivamente lo ejecutás).
"""

import os
import shutil
import sys

import environment  # noqa: F401  (efectos colaterales: HF_HOME, bundle de certificados, stdout UTF-8, ...)

OK, FAIL = "  OK  ", "FALLA "
problems = []


def check(title, fn):
    """Ejecuta un chequeo e imprime el resultado. fn devuelve un string con el detalle."""
    try:
        detail = fn()
    except Exception as e:
        problems.append(title)
        print(f"[{FAIL}] {title}")
        print(f"          {type(e).__name__}: {e}")
        return False
    print(f"[{OK}] {title}")
    if detail:
        for line in detail.splitlines():
            print(f"          {line}")
    return True


# -----------------------------------------------------------------------------

def check_python():
    v = sys.version_info
    if v < (3, 10):
        raise RuntimeError(f"Python {v.major}.{v.minor} es demasiado viejo, hace falta 3.10 o más nuevo")
    venv = sys.prefix
    warning = ""
    if sys.prefix == sys.base_prefix:
        warning = "\nADVERTENCIA: no parece haber ningún entorno virtual activo."
    return f"Python {v.major}.{v.minor}.{v.micro}\nintérprete: {sys.executable}\nvenv: {venv}{warning}"


def check_dependencies():
    missing = []
    versions = []
    for name in ("torch", "tokenizers", "datasets", "sklearn", "matplotlib", "tiktoken"):
        try:
            mod = __import__(name)
            versions.append(f"{name} {getattr(mod, '__version__', '?')}")
        except ImportError:
            missing.append(name)
    if missing:
        raise ImportError(
            f"faltan: {', '.join(missing)}. Ejecutá: pip install -r requirements.txt"
        )
    return "\n".join(versions)


def check_huggingface_connectivity():
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info("roneneldan/TinyStories")
    return (
        f"se llegó al Hub OK, el dataset '{info.id}' tiene {len(info.siblings)} archivos\n"
        "(esto fue solo una llamada de metadatos -- la descarga real ocurre en 00_dataset)"
    )


def check_environment():
    lines = []
    for k, v in environment.diagnostics().items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def check_ollama():
    path = shutil.which("ollama")
    if path is None:
        return (
            "no está instalado localmente -- es lo esperable fuera de Colab.\n"
            "La Etapa 5 (el juez) instala Ollama dentro de la propia VM de Colab;\n"
            "no hay nada que hacer acá salvo que planees correr esa etapa localmente."
        )
    return f"encontrado en {path}"


# -----------------------------------------------------------------------------

def main():
    print("Chequeando el entorno para la consigna del Laboratorio de Lenguaje Emergente\n")
    check("1. Versión de Python", check_python)
    ok_deps = check("2. Dependencias instaladas", check_dependencies)
    if ok_deps:
        check("3. Conectividad con el HuggingFace Hub", check_huggingface_connectivity)
    else:
        print(f"[{FAIL}] 3. Conectividad con el HuggingFace Hub")
        print("          salteado: faltan dependencias")
        problems.append("3. Conectividad con el HuggingFace Hub")
    check("4. Entorno / dispositivo", check_environment)
    check("5. Ollama (opcional, solo Etapa 5)", check_ollama)

    print()
    if problems:
        print(f"{len(problems)} chequeo(s) con problemas:")
        for p in problems:
            print(f"  - {p}")
        print("\nMirá la sección de Preparación del entorno en README.md antes de continuar.")
        return 1

    print("Está todo en orden. Ya podés empezar con 00_dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
