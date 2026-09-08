"""
Utilidades compartidas de carga de datasets, reutilizadas por la Etapa 0 (inspección) y la Etapa 4 (SFT).

TRAMPA MEDIDA, vale la pena leerla antes de tocar este archivo
-----------------------------------------------------------
`roneneldan/TinyStories-Instruct` guarda sus datos con UNA LÍNEA POR FILA, no un ejemplo de entrenamiento
por fila: un subconjunto aleatorio de líneas de encabezado `Features:` / `Words:` / `Summary:` / `Random sentence:`
(el orden no es fijo -- confirmado muestreando 5.000 filas), después una línea `Story:`, una línea en blanco, el
cuento mismo (a menudo repartido en varias líneas), y un delimitador `<|endoftext|>` que marca el final
de un registro. `load_instruct_records` reagrupa líneas consecutivas en registros completos.

También medido: `load_dataset(..., split="train[:N]")` sobre este dataset en particular NO hace una
lectura parcial barata -- es un dataset legacy con script de carga, así que `datasets` materializa el split
completo de ~21,7M de líneas antes de cortar. Ese es un costo único de ~20-30s en la primera corrida (después queda cacheado), no
un cuelgue. `roneneldan/TinyStories` (el corpus liso de preentrenamiento) no tiene este problema -- es
un dataset Parquet estándar y se corta en porciones de forma barata.

Una segunda razón, específica de Windows, para preferir `roneneldan/TinyStories-Instruct` por sobre el
mirror de la comunidad `skeskinen/TinyStories-Instruct-hf`: el script de carga legacy del mirror arma un archivo de lock
cuyo nombre incrusta la ruta completa y mutilada del directorio de caché. En una ruta de clonado profunda (un nombre
de usuario largo de Windows, o un repositorio anidado unos cuantos directorios adentro -- exactamente la situación de este repositorio), ese nombre
de archivo de lock excede lo que Windows acepta y `datasets` rompe con `WinError 206` antes
siquiera de empezar a descargar. `roneneldan/TinyStories-Instruct` no pasa por ese camino de código. Si alguna vez
necesitás cambiar de dataset, chequeá esto primero.
"""

from __future__ import annotations

from datasets import load_dataset

END_OF_RECORD = "<|endoftext|>"


def load_tinystories(split: str = "train", limit: int | None = None):
    """El corpus liso de preentrenamiento. Se corta en porciones de forma barata -- es un dataset Parquet estándar."""
    split_expr = f"{split}[:{limit}]" if limit else split
    return load_dataset("roneneldan/TinyStories", split=split_expr)


def load_instruct_records(split: str = "train", limit: int | None = None) -> list[str]:
    """
    Devuelve una lista de bloques completos de texto instrucción+cuento, cada uno listo para entrenar tal cual:

        Features: Dialogue, BadEnding
        Words: quit, oak, gloomy
        Story:

        Sara and Ben were playing in the park. ...

    `limit` corta después de juntar esa cantidad de registros completos -- no evita el costo único
    de generación descrito en el docstring de este módulo, que ocurre dentro de `load_dataset`
    mismo, antes de que haya ninguna fila disponible.
    """
    raw = load_dataset("roneneldan/TinyStories-Instruct", split=split)

    records: list[str] = []
    current: list[str] = []
    for row in raw:
        line = row["text"]
        if line.strip() == END_OF_RECORD:
            if current:
                records.append("\n".join(current).strip())
                current = []
            if limit and len(records) >= limit:
                break
        else:
            current.append(line)
    if current and not (limit and len(records) >= limit):
        records.append("\n".join(current).strip())
    return records
