import json
import unicodedata
from pathlib import Path

POSICIONES = ("portero", "defensa", "centrocampista", "delantero")
MINIMO_JUGADORES = 11


class ErrorPlantilla(Exception):
    pass


def cargar_plantilla(ruta: Path, jugadores: list[dict]) -> list[dict]:
    entradas = _leer(Path(ruta))
    if len(entradas) < MINIMO_JUGADORES:
        raise ErrorPlantilla(
            f"La plantilla de {ruta} tiene {len(entradas)} jugadores y hacen falta "
            f"al menos {MINIMO_JUGADORES}"
        )

    por_nombre = {}
    for jugador in jugadores:
        por_nombre.setdefault(_clave(jugador["nombre"]), []).append(jugador)

    plantilla = []
    vistos = set()
    for entrada in entradas:
        nombre, posicion = entrada["nombre"], entrada["posicion"]
        if posicion not in POSICIONES:
            raise ErrorPlantilla(
                f"La posición de {nombre} es {posicion!r} y debe ser una de: "
                + ", ".join(POSICIONES)
            )
        coincidencias = por_nombre.get(_clave(nombre), [])
        if not coincidencias:
            raise ErrorPlantilla(f"{nombre} no está en la lista oficial de jugadores")
        if len(coincidencias) > 1:
            raise ErrorPlantilla(
                f"{nombre} es ambiguo: coincide con {len(coincidencias)} jugadores oficiales"
            )
        oficial = coincidencias[0]
        if oficial["id"] in vistos:
            raise ErrorPlantilla(f"{nombre} aparece más de una vez en la plantilla")
        vistos.add(oficial["id"])
        plantilla.append(
            {
                "nombre": oficial["nombre"],
                "jugador_id": oficial["id"],
                "equipo": oficial["equipo"],
                "posicion": posicion,
            }
        )
    return plantilla


def _leer(ruta: Path) -> list[dict]:
    try:
        entradas = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ErrorPlantilla(f"No se puede leer la plantilla de {ruta}: {error}") from error
    if not isinstance(entradas, list) or any(
        not isinstance(entrada, dict)
        or not isinstance(entrada.get("nombre"), str)
        or not isinstance(entrada.get("posicion"), str)
        for entrada in entradas
    ):
        raise ErrorPlantilla(
            f"La plantilla de {ruta} debe ser una lista de objetos con nombre y posicion"
        )
    return entradas


def _clave(nombre: str) -> str:
    sin_marcas = "".join(
        caracter
        for caracter in unicodedata.normalize("NFKD", nombre)
        if not unicodedata.combining(caracter)
    )
    return " ".join(sin_marcas.casefold().split())
