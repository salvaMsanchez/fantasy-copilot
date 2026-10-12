import re
import unicodedata

MINIMO_CARACTERES = 500
MAXIMO_SIN_TROCEAR = 15_000
TAMANO_TROZO = 10_000
SOLAPE = 500

CAMPOS_VIDEO = ("canal", "video_id", "titulo", "fecha", "jornada", "temporada", "url")

_ETIQUETA = re.compile(r"\[[^\]]*\]")
_SIMBOLO = re.compile(r"[^\w\s.,;:¿?¡!%+€'’-]|_")
_APOSTROFO_O_GUION_SUELTO = re.compile(r"(?<![^\W_])['’-]|['’-](?![^\W_])")


def limpiar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFC", texto)
    texto = _ETIQUETA.sub(" ", texto)
    texto = _SIMBOLO.sub(" ", texto)
    texto = _APOSTROFO_O_GUION_SUELTO.sub(" ", texto)
    return " ".join(texto.split())


def preparar_textos(transcripciones: list[dict]) -> list[dict]:
    textos = []
    for transcripcion in transcripciones:
        if transcripcion["estado"] != "ok" or not transcripcion["transcripcion_cruda"]:
            continue
        texto = limpiar_texto(transcripcion["transcripcion_cruda"])
        if len(texto) < MINIMO_CARACTERES:
            continue
        video = {campo: transcripcion[campo] for campo in CAMPOS_VIDEO}
        if len(texto) <= MAXIMO_SIN_TROCEAR:
            textos.append({**video, "texto": texto, "chunk": None, "total_chunks": 1})
            continue
        trozos = _trocear(texto)
        textos.extend(
            {**video, "texto": trozo, "chunk": numero, "total_chunks": len(trozos)}
            for numero, trozo in enumerate(trozos, 1)
        )
    return textos


def _trocear(texto: str) -> list[str]:
    palabras = texto.split(" ")
    trozos = []
    inicio = fin = 0
    while fin < len(palabras):
        # Cada trozo añade al menos una palabra nueva al solape, para avanzar siempre.
        fin += 1
        longitud = len(" ".join(palabras[inicio:fin]))
        while fin < len(palabras) and longitud + 1 + len(palabras[fin]) <= TAMANO_TROZO:
            longitud += 1 + len(palabras[fin])
            fin += 1
        trozos.append(" ".join(palabras[inicio:fin]))

        anterior, inicio, solape = inicio, fin, -1
        while inicio > anterior and solape + 1 + len(palabras[inicio - 1]) <= SOLAPE:
            inicio -= 1
            solape += 1 + len(palabras[inicio])
    return trozos
