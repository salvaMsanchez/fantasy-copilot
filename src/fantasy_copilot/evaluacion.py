import json
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from openai import OpenAIError

from fantasy_copilot.insights import ACCIONES, extraer_insights
from fantasy_copilot.normalizacion import resolver_nombre
from fantasy_copilot.plantilla import clave_nombre
from fantasy_copilot.textos import CAMPOS_VIDEO

CAMPOS_TEXTO = (*CAMPOS_VIDEO, "chunk", "texto")


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


def evaluar_extraccion(*, datos_dir: Path, modelo: str, cliente, esperar=time.sleep) -> dict:
    fichero = Path(datos_dir) / "evaluacion" / "extraccion.json"
    textos = _leer_conjunto_extraccion(fichero)

    detalle = []
    # Cada texto se extrae en su propia carpeta temporal: así se ignora lo ya procesado en
    # la jornada y no se escribe nada en ella.
    with tempfile.TemporaryDirectory() as temporal:
        for posicion, texto in enumerate(textos):
            entrada = {campo: texto[campo] for campo in CAMPOS_TEXTO}
            resultado = extraer_insights(
                [entrada],
                jornada=texto["jornada"],
                temporada=texto["temporada"],
                datos_dir=Path(temporal) / str(posicion),
                modelo=modelo,
                cliente=cliente,
                esperar=esperar,
            )
            esperados = _recuento_insights(texto["insights_esperados"])
            extraidos = _recuento_insights(resultado["insights"])
            correctos = sum((esperados & extraidos).values())
            detalle.append(
                {
                    "video_id": texto["video_id"],
                    "chunk": texto["chunk"],
                    "correctos": correctos,
                    "omitidos": esperados.total() - correctos,
                    "inventados": extraidos.total() - correctos,
                    "fallido": bool(resultado["fallidos"]),
                }
            )

    correctos, omitidos, inventados = (
        sum(item[clave] for item in detalle) for clave in ("correctos", "omitidos", "inventados")
    )
    return {
        "correctos": correctos,
        "omitidos": omitidos,
        "inventados": inventados,
        "fallidos": sum(item["fallido"] for item in detalle),
        "precision": _fraccion(correctos, correctos + inventados),
        "cobertura": _fraccion(correctos, correctos + omitidos),
        "textos": detalle,
    }


def _recuento_insights(insights: list[dict]) -> Counter:
    return Counter((clave_nombre(insight["jugador"]), insight["accion"]) for insight in insights)


def _leer_json(fichero: Path):
    if not fichero.exists():
        raise ErrorConjunto(f"No existe el conjunto etiquetado {fichero}")
    try:
        return json.loads(fichero.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ErrorConjunto(f"El conjunto {fichero} no es JSON válido: {error}") from error


def _leer_conjunto_extraccion(fichero: Path) -> list[dict]:
    contenido = _leer_json(fichero)
    textos = contenido.get("textos") if isinstance(contenido, dict) else None
    if not isinstance(textos, list):
        raise ErrorConjunto(f"Al conjunto {fichero} le falta la lista de textos")
    for posicion, texto in enumerate(textos, start=1):
        if not isinstance(texto, dict) or any(campo not in texto for campo in CAMPOS_TEXTO):
            raise ErrorConjunto(
                f"Al texto {posicion} del conjunto {fichero} le falta alguno de los campos "
                f"{', '.join(CAMPOS_TEXTO)}"
            )
        esperados = texto.get("insights_esperados")
        if not isinstance(esperados, list) or not all(
            isinstance(esperado, dict)
            and isinstance(esperado.get("jugador"), str)
            and esperado.get("accion") in ACCIONES
            for esperado in esperados
        ):
            raise ErrorConjunto(
                f"El texto {posicion} del conjunto {fichero} necesita `insights_esperados`, "
                "una lista de insights con `jugador` y `accion` (alinear o evitar)"
            )
    return textos


def _leer_conjunto(fichero: Path) -> tuple[int, list[dict]]:
    contenido = _leer_json(fichero)
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
