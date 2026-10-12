import json
import re
import time
from collections.abc import Callable
from pathlib import Path

import requests
from bs4 import BeautifulSoup

_BASE = "https://www.transfermarkt.es"
_CABECERAS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}
_PERFIL_JUGADOR = re.compile(r"/profil/spieler/(\d+)$")
_SEGUNDOS_ENTRE_EQUIPOS = 1
_SEGUNDOS_DE_ESPERA_MAXIMA = 30


class ErrorTransfermarkt(Exception):
    pass


def extraer_jugadores(html: str, equipo: str) -> list[dict]:
    pagina = BeautifulSoup(html, "html.parser")
    jugadores = {}
    for fila in pagina.select("table.items > tbody > tr"):
        enlace = fila.select_one("td.hauptlink a[href]")
        perfil = _PERFIL_JUGADOR.search(enlace["href"]) if enlace else None
        if not perfil or perfil.group(1) in jugadores:
            continue
        celda_posicion = fila.select_one("td.posrela table tr:nth-of-type(2) td")
        posicion = celda_posicion.get_text(strip=True) if celda_posicion else ""
        jugadores[perfil.group(1)] = {
            "id": perfil.group(1),
            "nombre": enlace.get_text(strip=True),
            "posicion": posicion or None,
            "equipo": equipo,
        }
    return list(jugadores.values())


def obtener_jugadores(
    temporada: int,
    *,
    datos_dir: Path,
    cliente=requests,
    esperar: Callable[[float], None] = time.sleep,
) -> list[dict]:
    ruta = Path(datos_dir) / f"jugadores_{temporada}.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))

    url_liga = f"{_BASE}/laliga/startseite/wettbewerb/ES1/plus/?saison_id={temporada}"
    equipos = _extraer_equipos(_descargar(cliente, url_liga, "LaLiga"))
    if not equipos:
        raise ErrorTransfermarkt("La página de LaLiga no contiene ningún equipo")

    por_id = {}
    for indice, (equipo, url_plantilla) in enumerate(equipos):
        if indice:
            esperar(_SEGUNDOS_ENTRE_EQUIPOS)
        plantilla = extraer_jugadores(_descargar(cliente, url_plantilla, equipo), equipo)
        if not plantilla:
            raise ErrorTransfermarkt(f"La página de {equipo} no contiene ningún jugador")
        for jugador in plantilla:
            por_id.setdefault(jugador["id"], {**jugador, "equipos": []})["equipos"].append(equipo)
    jugadores = list(por_id.values())

    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(jugadores, ensure_ascii=False, indent=2), encoding="utf-8")
    return jugadores


def _extraer_equipos(html: str) -> list[tuple[str, str]]:
    pagina = BeautifulSoup(html, "html.parser")
    equipos = {}
    for enlace in pagina.select("table.items > tbody > tr td.hauptlink a[href]"):
        ruta = enlace["href"]
        nombre = enlace.get_text(strip=True)
        if nombre and "/startseite/verein/" in ruta:
            equipos.setdefault(_BASE + ruta.replace("/startseite/", "/kader/"), nombre)
    return [(nombre, url) for url, nombre in equipos.items()]


def _descargar(cliente, url: str, nombre: str) -> str:
    try:
        respuesta = cliente.get(url, headers=_CABECERAS, timeout=_SEGUNDOS_DE_ESPERA_MAXIMA)
    except OSError as error:
        raise ErrorTransfermarkt(f"No se pudo descargar la página de {nombre}: {error}") from error
    if respuesta.status_code != 200:
        raise ErrorTransfermarkt(
            f"La página de {nombre} respondió con el código {respuesta.status_code}"
        )
    return respuesta.text
