import json
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from openai import OpenAIError

from fantasy_copilot.agente import PREGUNTA, TAMANO_ONCE, TOP_K, recomendar_once
from fantasy_copilot.indice import identificador
from fantasy_copilot.insights import ACCIONES, extraer_insights
from fantasy_copilot.normalizacion import resolver_nombre
from fantasy_copilot.plantilla import POSICIONES, ErrorPlantilla, clave_nombre
from fantasy_copilot.puntos import ErrorPuntos
from fantasy_copilot.textos import CAMPOS_VIDEO

CAMPOS_TEXTO = (*CAMPOS_VIDEO, "chunk", "texto")
# Jugadores por posición, en el orden de POSICIONES.
FORMACIONES = ((1, 4, 3, 3), (1, 4, 4, 2))


class ErrorConjunto(Exception):
    pass


class ErrorJornadaSinIngerir(Exception):
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


def evaluar_recuperacion(
    plantilla: list[dict], *, jornada: int, temporada: int, datos_dir: Path, indice
) -> dict:
    fichero = Path(datos_dir) / str(temporada) / f"j{jornada:02d}" / "insights_normalizados.json"
    if not fichero.exists():
        raise ErrorJornadaSinIngerir(
            f"No hay datos de la jornada {jornada} de la temporada {temporada}: ejecuta antes "
            f"fantasy-copilot ingesta --jornada {jornada} --temporada {temporada}"
        )
    insights = json.loads(fichero.read_text(encoding="utf-8"))
    indice.indexar(insights)

    detalle = []
    for jugador in plantilla:
        relevantes = {
            identificador(insight)
            for insight in insights
            if insight["jugador_id"] == jugador["jugador_id"]
        }
        if not relevantes:
            detalle.append(
                {
                    "jugador": jugador["nombre"],
                    "jugador_id": jugador["jugador_id"],
                    "precision": None,
                    "cobertura": None,
                    "sin_insights": True,
                }
            )
            continue
        recuperados = [
            resultado["id"]
            for resultado in indice.consultar(
                PREGUNTA.format(nombre=jugador["nombre"]),
                jornada=jornada,
                temporada=temporada,
                top_k=TOP_K,
            )
        ]
        detalle.append(
            {
                "jugador": jugador["nombre"],
                "jugador_id": jugador["jugador_id"],
                "precision": _fraccion(
                    sum(id_ in relevantes for id_ in recuperados), len(recuperados)
                ),
                "cobertura": _fraccion(len(relevantes & set(recuperados)), len(relevantes)),
                "sin_insights": False,
            }
        )

    return {
        "precision": _media(jugador["precision"] for jugador in detalle),
        "cobertura": _media(jugador["cobertura"] for jugador in detalle),
        "jugadores": detalle,
    }


def evaluar_once(
    plantilla: list[dict],
    *,
    jornada: int,
    temporada: int,
    datos_dir: Path,
    indice,
    modelo: str,
    cliente,
    esperar=time.sleep,
) -> dict:
    if not _formaciones_posibles(plantilla):
        raise ErrorPlantilla(
            "La plantilla no permite ninguna formación válida "
            "(1 portero, 4 defensas, 3 o 4 centrocampistas y 2 o 3 delanteros)"
        )
    puntos = _leer_puntos(Path(datos_dir), jornada, temporada)
    if puntos is None:
        raise ErrorPuntos(
            f"No hay puntos guardados de la jornada {jornada} de la temporada {temporada}"
        )

    once = recomendar_once(
        plantilla,
        jornada=jornada,
        temporada=temporada,
        indice=indice,
        modelo=modelo,
        cliente=cliente,
        esperar=esperar,
    )["recomendacion"]["once_recomendado"]
    nombres = [jugador["jugador"] for jugador in once]
    por_nombre = {jugador["nombre"]: jugador["jugador_id"] for jugador in plantilla}
    elegidos = [por_nombre[nombre] for nombre in dict.fromkeys(nombres) if nombre in por_nombre]
    recomendado = {
        "jugadores": elegidos,
        "puntos": _suma(elegidos, puntos),
        "valido": len(nombres) == TAMANO_ONCE and len(elegidos) == TAMANO_ONCE,
    }

    mejor = _mejor_once(plantilla, puntos)
    mejor["puntos"] = _suma(mejor["jugadores"], puntos)

    anteriores = [
        previos
        for previa in range(1, jornada)
        if (previos := _leer_puntos(Path(datos_dir), previa, temporada)) is not None
    ]
    eleccion_simple = None
    if anteriores:
        medias = {
            jugador["jugador_id"]: sum(p.get(jugador["jugador_id"], 0) for p in anteriores)
            / len(anteriores)
            for jugador in plantilla
        }
        eleccion_simple = _mejor_once(plantilla, medias)
        eleccion_simple["puntos"] = _suma(eleccion_simple["jugadores"], puntos)

    return {
        "recomendado": recomendado,
        "mejor_once": mejor,
        "eleccion_simple": eleccion_simple,
        "proporcion": recomendado["puntos"] / mejor["puntos"] if mejor["puntos"] > 0 else None,
    }


def _formaciones_posibles(plantilla: list[dict]) -> list[tuple[int, ...]]:
    disponibles = Counter(jugador["posicion"] for jugador in plantilla)
    return [
        formacion
        for formacion in FORMACIONES
        if all(
            disponibles[posicion] >= n for posicion, n in zip(POSICIONES, formacion, strict=True)
        )
    ]


def _mejor_once(plantilla: list[dict], valor: dict[str, float]) -> dict:
    # El orden estable de sorted deja antes, en los empates, al que va antes en la plantilla.
    ordenados = sorted(plantilla, key=lambda jugador: -valor.get(jugador["jugador_id"], 0))
    candidatos = []
    for formacion in _formaciones_posibles(plantilla):
        jugadores = [
            jugador["jugador_id"]
            for posicion, n in zip(POSICIONES, formacion, strict=True)
            for jugador in [j for j in ordenados if j["posicion"] == posicion][:n]
        ]
        candidatos.append((sum(valor.get(i, 0) for i in jugadores), formacion, jugadores))
    _, formacion, jugadores = max(candidatos, key=lambda candidato: candidato[0])
    return {"formacion": "-".join(map(str, formacion)), "jugadores": jugadores}


def _leer_puntos(datos_dir: Path, jornada: int, temporada: int) -> dict[str, int] | None:
    fichero = datos_dir / str(temporada) / f"j{jornada:02d}" / "puntos.json"
    if not fichero.exists():
        return None
    contenido = json.loads(fichero.read_text(encoding="utf-8"))
    return {fila["jugador_id"]: fila["puntos"] for fila in contenido["puntos"]}


def _suma(jugadores: list[str], puntos: dict[str, int]) -> int:
    return sum(puntos.get(jugador_id, 0) for jugador_id in jugadores)


def _media(valores) -> float | None:
    presentes = [valor for valor in valores if valor is not None]
    return sum(presentes) / len(presentes) if presentes else None


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
