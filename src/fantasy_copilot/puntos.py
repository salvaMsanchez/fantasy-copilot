import json
import re
import unicodedata
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

_URL = "https://www.futbolfantasy.com/laliga/puntos/{anio}/{jornada}/laliga-fantasy"
_CABECERAS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}
_SEGUNDOS_DE_ESPERA_MAXIMA = 30
_DIAS_HASTA_DEFINITIVOS = 3
_HORA_ESPANOLA = ZoneInfo("Europe/Madrid")
_MESES = {
    nombre: numero
    for numero, nombre in enumerate(
        (
            "enero febrero marzo abril mayo junio julio "
            "agosto septiembre octubre noviembre diciembre"
        ).split(),
        start=1,
    )
}
_TITULO_EQUIPO = re.compile(r"Puntos\s+(?:de la|del|de)\s+(.+)", re.IGNORECASE)
_FECHA_PARTIDO = re.compile(r"(\d{1,2}) de (\w+) del? (\d{4}) a las (\d{1,2}):(\d{2})")
_FICHA = re.compile(r"/jugadores/([^/]+)")
_TRANSLITERACION = str.maketrans({"ø": "o", "æ": "ae", "ß": "ss", "ł": "l", "đ": "d", "ı": "i"})


class ErrorPuntos(Exception):
    pass


def obtener_puntos(
    jornada: int,
    temporada: int,
    *,
    jugadores: list[dict],
    datos_dir: Path,
    cliente=requests,
    ahora: datetime,
) -> dict:
    fichero = Path(datos_dir) / str(temporada) / f"j{jornada:02d}" / "puntos.json"
    if fichero.exists():
        return {**json.loads(fichero.read_text(encoding="utf-8")), "provisional": False}

    nombre_jornada = f"la jornada {jornada} de la temporada {temporada}"
    html = _descargar(cliente, _URL.format(anio=temporada + 1, jornada=jornada), nombre_jornada)
    pagina = BeautifulSoup(html, "html.parser")

    filas = _extraer_filas(pagina)
    if not filas:
        raise ErrorPuntos(f"La página de {nombre_jornada} no tiene ningún jugador")
    inicios = _extraer_inicios(pagina)
    if not inicios:
        raise ErrorPuntos(f"La página de {nombre_jornada} no trae la fecha de ningún partido")

    equipos = _equipos_oficiales({equipo for equipo, _, _, _ in filas}, jugadores)
    puntos, sin_asociar = [], []
    for equipo, ficha, nombre_corto, valor in filas:
        jugador = _asociar(ficha, nombre_corto, equipos[equipo], jugadores)
        if jugador:
            puntos.append(
                {"jugador_id": jugador["id"], "nombre": jugador["nombre"], "puntos": valor}
            )
        else:
            sin_asociar.append(
                {"nombre": ficha.replace("-", " ").strip(), "equipo": equipo, "puntos": valor}
            )
    resultado = {"puntos": puntos, "sin_asociar": sin_asociar}

    transcurrido = ahora.astimezone(UTC) - max(inicios).astimezone(UTC)
    provisional = transcurrido < timedelta(days=_DIAS_HASTA_DEFINITIVOS)
    if not provisional:
        fichero.parent.mkdir(parents=True, exist_ok=True)
        fichero.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**resultado, "provisional": provisional}


def _extraer_filas(pagina: BeautifulSoup) -> list[tuple[str, str, str, int]]:
    filas = []
    equipo = None
    for elemento in pagina.find_all(["h2", "tr"]):
        if elemento.name == "h2":
            titulo = _TITULO_EQUIPO.search(" ".join(elemento.get_text().split()))
            if titulo:
                equipo = titulo.group(1).strip()
            continue
        if equipo is None or "plegable" not in (elemento.get("class") or []):
            continue
        casilla = elemento.select_one("span.laliga-fantasy")
        desglose = elemento.find_next_sibling("tr")
        enlace = desglose.select_one('a[href*="/jugadores/"]') if desglose else None
        ficha = _FICHA.search(enlace["href"]) if enlace else None
        if casilla is None or ficha is None:
            continue
        texto = casilla.get_text(strip=True)
        valor = int(texto) if re.fullmatch(r"-?\d+", texto) else 0
        filas.append((equipo, ficha.group(1), _nombre_corto(elemento), valor))
    return filas


def _nombre_corto(fila) -> str:
    celda = fila.select_one("td.name")
    if celda is None:
        return ""
    # Solo el texto propio de la celda: el minuto de un cambio va en un span aparte.
    return " ".join(celda.find_all(string=True, recursive=False)).strip()


def _extraer_inicios(pagina: BeautifulSoup) -> list[datetime]:
    inicios = []
    for fecha in pagina.select("div.fecha"):
        partes = _FECHA_PARTIDO.search(fecha.get_text(" ", strip=True).lower())
        if not partes or partes.group(2) not in _MESES:
            continue
        dia, mes, anio, hora, minuto = partes.groups()
        inicios.append(
            datetime(
                int(anio), _MESES[mes], int(dia), int(hora), int(minuto), tzinfo=_HORA_ESPANOLA
            )
        )
    return inicios


def _equipos_oficiales(de_la_pagina: set[str], jugadores: list[dict]) -> dict[str, set[str]]:
    oficiales = {equipo for j in jugadores for equipo in j.get("equipos") or [j["equipo"]]}
    encajan = {
        corto: {o for o in oficiales if set(_palabras(corto)) <= set(_palabras(o))}
        for corto in de_la_pagina
    }
    # Un nombre corto puede encajar con varios equipos ("Deportivo" con el Dépor y el Alavés);
    # se descartan los que otro bloque de la página ya identifica sin duda ("Alavés").
    identificados = {next(iter(o)) for o in encajan.values() if len(o) == 1}
    return {
        corto: (o - identificados) if len(o) > 1 and o - identificados else o
        for corto, o in encajan.items()
    }


def _asociar(
    ficha: str, nombre_corto: str, equipos: set[str], jugadores: list[dict]
) -> dict | None:
    candidatos = [j for j in jugadores if equipos & set(j.get("equipos") or [j["equipo"]])]
    # Las fichas pueden acabar en un número que distingue a homónimos ("jesus-martinez-1").
    palabras_ficha = [palabra for palabra in _palabras(ficha) if not palabra.isdigit()]
    exactos = [
        j for j in candidatos if _mismas_palabras(palabras_ficha, _palabras(j["nombre"], True))
    ]
    if exactos:
        return exactos[0] if len(exactos) == 1 else None
    # Una ficha con un nombre de más ("isco-alarcon") o de menos ("terrats") solo vale si el
    # nombre corto de la página está en el nombre oficial ("Céspedes" descarta a "Andrés").
    palabras_cortas = _palabras(nombre_corto)
    if not palabras_cortas:
        return None
    parciales = [
        j
        for j in candidatos
        if _una_contiene_a_otra(palabras_ficha, _palabras(j["nombre"], True))
        and _contenidas(palabras_cortas, _palabras(j["nombre"], True))
    ]
    return parciales[0] if len(parciales) == 1 else None


def _una_contiene_a_otra(de_ficha: list[str], oficiales: list[str]) -> bool:
    return bool(de_ficha and oficiales) and (
        _contenidas(de_ficha, oficiales)
        or all(any(_misma_palabra(f, o) for f in de_ficha) for o in oficiales)
    )


def _contenidas(de_ficha: list[str], oficiales: list[str]) -> bool:
    return all(any(_misma_palabra(f, o) for o in oficiales) for f in de_ficha)


def _mismas_palabras(de_ficha: list[str], oficiales: list[str]) -> bool:
    return len(de_ficha) == len(oficiales) and all(map(_misma_palabra, de_ficha, oficiales))


def _misma_palabra(de_ficha: str, oficial: str) -> bool:
    """Las fichas de futbolfantasy pierden a veces una letra acentuada
    ("lvaro" por Álvaro, "gler" por Güler); otra letra de menos no vale."""
    if _sin_tildes(oficial) == de_ficha:
        return True
    return any(
        not letra.isascii() and _sin_tildes(oficial[:i] + oficial[i + 1 :]) == de_ficha
        for i, letra in enumerate(oficial)
    )


def _palabras(texto: str, con_tildes: bool = False) -> list[str]:
    palabras = re.findall(r"[^\W_]+", texto.lower().translate(_TRANSLITERACION))
    return palabras if con_tildes else [_sin_tildes(palabra) for palabra in palabras]


def _sin_tildes(palabra: str) -> str:
    return "".join(
        caracter
        for caracter in unicodedata.normalize("NFKD", palabra)
        if not unicodedata.combining(caracter)
    )


def _descargar(cliente, url: str, nombre_jornada: str) -> str:
    try:
        respuesta = cliente.get(url, headers=_CABECERAS, timeout=_SEGUNDOS_DE_ESPERA_MAXIMA)
    except OSError as error:
        raise ErrorPuntos(f"No se pudo descargar la página de {nombre_jornada}: {error}") from error
    if respuesta.status_code != 200:
        raise ErrorPuntos(
            f"La página de {nombre_jornada} respondió con el código {respuesta.status_code}"
        )
    return respuesta.text
