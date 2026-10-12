import re

ULTIMA_JORNADA = 38

_PATRON_JORNADA = re.compile(r"\b(?:jornada\s*|j)(\d+)", re.IGNORECASE)


def extraer_numero_jornada(titulo: str) -> int | None:
    for coincidencia in _PATRON_JORNADA.finditer(titulo):
        jornada = int(coincidencia.group(1))
        if 1 <= jornada <= ULTIMA_JORNADA:
            return jornada
    return None


def es_video_relevante(titulo: str) -> bool:
    if "mundial" in titulo.lower():
        return False
    return extraer_numero_jornada(titulo) is not None
