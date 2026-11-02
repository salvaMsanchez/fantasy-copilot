import asyncio
import json
import math
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from openai import OpenAIError

from fantasy_copilot.agente import PREGUNTA, TAMANO_ONCE, TOP_K, recomendar_once
from fantasy_copilot.indice import identificador
from fantasy_copilot.insights import ACCIONES, CRITERIOS, extraer_insights
from fantasy_copilot.normalizacion import resolver_nombre
from fantasy_copilot.plantilla import POSICIONES, ErrorPlantilla, clave_nombre
from fantasy_copilot.puntos import ErrorPuntos
from fantasy_copilot.textos import CAMPOS_VIDEO

CAMPOS_TEXTO = (*CAMPOS_VIDEO, "chunk", "texto")
# Jugadores por posición, en el orden de POSICIONES.
FORMACIONES = ((1, 4, 3, 3), (1, 4, 4, 2))
MAXIMO_TOKENS_JUEZ = 4096
PREGUNTA_INSIGHT = "¿Por qué recomienda el analista {accion} a {jugador}?"


CONJUNTOS = ("nombres", "extraccion", "recomendacion")


class ErrorConjunto(Exception):
    pass


class ErrorJornadaSinIngerir(Exception):
    pass


def evaluar_nombres(
    *, datos_dir: Path, jugadores_de: Callable[[int], list[dict]], modelo: str, cliente
) -> dict:
    fichero = fichero_conjunto(datos_dir, "nombres")
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
    fichero = fichero_conjunto(datos_dir, "extraccion")
    textos = _leer_conjunto_extraccion(fichero)

    detalle = []
    extraidos_con_texto = []
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
            extraidos_con_texto += [
                {"insight": extraido, "texto": texto["texto"]} for extraido in resultado["insights"]
            ]
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
        "insights": extraidos_con_texto,
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
    recomendacion: dict | None = None,
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

    if recomendacion is None:
        recomendacion = recomendar_once(
            plantilla,
            jornada=jornada,
            temporada=temporada,
            indice=indice,
            modelo=modelo,
            cliente=cliente,
            esperar=esperar,
        )
    once = recomendacion["recomendacion"]["once_recomendado"]
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


def crear_juez(modelo: str, cliente_asincrono):
    from ragas.llms import llm_factory

    juez = llm_factory(modelo, client=cliente_asincrono)
    # RAGAS 0.4.3 manda max_tokens a modelos como gpt-5.4-mini, que solo aceptan
    # max_completion_tokens: no reconoce las versiones con punto como de razonamiento.
    juez.model_args.pop("max_tokens", None)
    juez.model_args["max_completion_tokens"] = MAXIMO_TOKENS_JUEZ
    return juez


def evaluar_fidelidad(recomendacion: dict, *, juez) -> dict:
    jugadores = asyncio.run(
        _juzgar_resumenes(recomendacion["resumenes"], recomendacion["insights"], juez)
    )
    return {
        "media": _media(jugador["fidelidad"] for jugador in jugadores),
        "jugadores": jugadores,
    }


async def _juzgar_resumenes(resumenes: list[dict], contextos: list[list[dict]], juez) -> list:
    from ragas.metrics.collections import Faithfulness

    anotador = _juez_que_anota(juez)
    metrica = Faithfulness(llm=anotador)
    jugadores = []
    for resumen, contexto in zip(resumenes, contextos, strict=True):
        detalle = {"jugador": resumen["jugador"], "resumen": resumen["resumen"]}
        jugadores.append(detalle)
        if (
            not contexto
            or not resumen["resumen"]
            or resumen.get("accion_recomendada") == "sin_datos"
        ):
            detalle.update(fidelidad=None, no_respaldadas=[], estado="sin_datos", error=None)
            continue
        await _juzgar(
            metrica,
            anotador,
            detalle,
            pregunta=PREGUNTA.format(nombre=resumen["jugador"]),
            respuesta=resumen["resumen"],
            contextos=[json.dumps(i, ensure_ascii=False) for i in contexto],
        )
    return jugadores


def evaluar_fidelidad_insights(extraccion: dict, *, juez) -> dict:
    insights = asyncio.run(_juzgar_insights(extraccion["insights"], juez))
    return {"media": _media(insight["fidelidad"] for insight in insights), "insights": insights}


async def _juzgar_insights(extraidos: list[dict], juez) -> list:
    from ragas.metrics.collections import Faithfulness

    anotador = _juez_que_anota(juez)
    metrica = Faithfulness(llm=anotador)
    insights = []
    for extraido in extraidos:
        insight = extraido["insight"]
        detalle = {
            "video_id": insight["fuente"]["video_id"],
            "chunk": insight["fuente"]["chunk"],
            "jugador": insight["jugador"],
        }
        insights.append(detalle)
        razonamiento = [insight["razon_corta"]] + [
            insight["criterios"][criterio]
            for criterio in CRITERIOS
            if insight["criterios"].get(criterio)
        ]
        await _juzgar(
            metrica,
            anotador,
            detalle,
            pregunta=PREGUNTA_INSIGHT.format(accion=insight["accion"], jugador=insight["jugador"]),
            respuesta="\n".join(razonamiento),
            contextos=[extraido["texto"]],
        )
    return insights


async def _juzgar(metrica, anotador, detalle: dict, *, pregunta, respuesta, contextos) -> None:
    detalle.update(fidelidad=None, no_respaldadas=[], estado="juzgado", error=None)
    anotador.veredictos = None
    try:
        resultado = await metrica.ascore(
            user_input=pregunta, response=respuesta, retrieved_contexts=contextos
        )
    # El juez pasa por instructor y OpenAI, que lanzan excepciones de varios tipos.
    except Exception as error:
        detalle["estado"] = "error"
        detalle["error"] = f"{type(error).__name__}: {error}"
        return
    if not math.isnan(resultado.value):
        detalle["fidelidad"] = resultado.value
    if anotador.veredictos is not None:
        detalle["no_respaldadas"] = [
            {"afirmacion": veredicto.statement, "motivo": veredicto.reason}
            for veredicto in anotador.veredictos.statements
            if not veredicto.verdict
        ]


def _juez_que_anota(juez):
    """Envuelve al juez para guardar sus veredictos, que la métrica no devuelve."""
    from ragas.llms.base import InstructorBaseRagasLLM
    from ragas.metrics.collections.faithfulness.util import NLIStatementOutput

    class JuezQueAnota(InstructorBaseRagasLLM):
        def __init__(self):
            self.veredictos = None

        def generate(self, prompt, response_model):
            return juez.generate(prompt, response_model)

        async def agenerate(self, prompt, response_model):
            respuesta = await juez.agenerate(prompt, response_model)
            if response_model is NLIStatementOutput:
                self.veredictos = respuesta
            return respuesta

    return JuezQueAnota()


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


def fichero_conjunto(datos_dir: Path, conjunto: str) -> Path:
    return Path(datos_dir) / "evaluacion" / f"{conjunto}.json"


def leer_conjunto(datos_dir: Path, conjunto: str):
    """Lee y valida un conjunto etiquetado; lanza ErrorConjunto si está dañado."""
    fichero = fichero_conjunto(datos_dir, conjunto)
    if conjunto == "nombres":
        return _leer_conjunto(fichero)[1]
    if conjunto == "extraccion":
        return _leer_conjunto_extraccion(fichero)
    return _leer_conjunto_recomendacion(fichero)


def _leer_conjunto_recomendacion(fichero: Path) -> list[dict]:
    contenido = _leer_json(fichero)
    casos = contenido.get("casos") if isinstance(contenido, dict) else None
    if not isinstance(casos, list):
        raise ErrorConjunto(f"Al conjunto {fichero} le falta la lista de casos")
    for posicion, caso in enumerate(casos, start=1):
        if (
            not isinstance(caso, dict)
            or not isinstance(caso.get("temporada"), int)
            or not isinstance(caso.get("jornada"), int)
            or caso["jornada"] < 1
            or not isinstance(caso.get("plantilla"), list)
        ):
            raise ErrorConjunto(
                f"El caso {posicion} del conjunto {fichero} necesita `temporada`, "
                "`jornada` y `plantilla`"
            )
    return casos


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
