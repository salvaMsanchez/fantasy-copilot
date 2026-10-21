import json
from collections.abc import Callable
from pathlib import Path

from openai import OpenAIError

from fantasy_copilot.normalizacion import resolver_nombre


class ErrorConjunto(Exception):
    pass


def evaluar_nombres(
    *, datos_dir: Path, jugadores_de: Callable[[int], list[dict]], modelo: str, cliente
) -> dict:
    fichero = Path(datos_dir) / "evaluacion" / "nombres.json"
    temporada, conjunto = _leer_conjunto(fichero)
    jugadores = jugadores_de(temporada)
    ids_oficiales = {jugador["id"] for jugador in jugadores}
    for caso in conjunto:
        esperado = caso["jugador_id"]
        if esperado is not None and esperado not in ids_oficiales:
            raise ErrorConjunto(
                f"El jugador esperado de «{caso['nombre_transcrito']}» ({esperado}) no está "
                f"en la lista oficial de la temporada {temporada}"
            )

    detalle = []
    for caso in conjunto:
        try:
            obtenido = resolver_nombre(
                caso["nombre_transcrito"],
                caso.get("cita") or "",
                jugadores,
                modelo=modelo,
                cliente=cliente,
            )
        except (OpenAIError, ValueError, TypeError, KeyError):
            obtenido = None
        detalle.append(
            {
                "nombre_transcrito": caso["nombre_transcrito"],
                "jugador_id_esperado": caso["jugador_id"],
                "jugador_id_obtenido": obtenido if obtenido in ids_oficiales else None,
            }
        )

    resueltos = [n for n in detalle if n["jugador_id_obtenido"] is not None]
    con_jugador = [n for n in detalle if n["jugador_id_esperado"] is not None]
    correctos = [n for n in resueltos if n["jugador_id_obtenido"] == n["jugador_id_esperado"]]
    return {
        "precision": _fraccion(len(correctos), len(resueltos)),
        "cobertura": _fraccion(len(correctos), len(con_jugador)),
        "porcentaje_eliminados": _fraccion(len(detalle) - len(resueltos), len(detalle)),
        "nombres": detalle,
    }


def _leer_conjunto(fichero: Path) -> tuple[int, list[dict]]:
    if not fichero.exists():
        raise ErrorConjunto(f"No existe el conjunto etiquetado {fichero}")
    try:
        contenido = json.loads(fichero.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ErrorConjunto(f"El conjunto {fichero} no es JSON válido: {error}") from error
    if not isinstance(contenido, dict) or not isinstance(contenido.get("temporada"), int):
        raise ErrorConjunto(f"Al conjunto {fichero} le falta la temporada")
    nombres = contenido.get("nombres")
    if not isinstance(nombres, list):
        raise ErrorConjunto(f"Al conjunto {fichero} le falta la lista de nombres")
    for posicion, caso in enumerate(nombres, start=1):
        if (
            not isinstance(caso, dict)
            or not isinstance(caso.get("nombre_transcrito"), str)
            or "jugador_id" not in caso
        ):
            raise ErrorConjunto(
                f"El nombre {posicion} del conjunto {fichero} necesita "
                "`nombre_transcrito` y `jugador_id`"
            )
    return contenido["temporada"], nombres


def _fraccion(parte: int, total: int) -> float | None:
    return parte / total if total else None
