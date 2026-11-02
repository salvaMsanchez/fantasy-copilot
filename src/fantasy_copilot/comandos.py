import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAIError

from fantasy_copilot import informe
from fantasy_copilot.agente import ErrorRecomendacion, recomendar_once
from fantasy_copilot.configuracion import ErrorConfiguracion, cargar_configuracion
from fantasy_copilot.evaluacion import (
    CONJUNTOS,
    ErrorConjunto,
    ErrorJornadaSinIngerir,
    evaluar_extraccion,
    evaluar_fidelidad,
    evaluar_fidelidad_insights,
    evaluar_nombres,
    evaluar_once,
    evaluar_recuperacion,
    fichero_conjunto,
    leer_conjunto,
)
from fantasy_copilot.indice import ErrorInsightNoNormalizado, Indice
from fantasy_copilot.insights import ErrorInsightsGuardados, extraer_insights
from fantasy_copilot.jugadores import ErrorTransfermarkt, obtener_jugadores
from fantasy_copilot.normalizacion import ErrorDecisionesGuardadas, normalizar_insights
from fantasy_copilot.plantilla import ErrorPlantilla, cargar_plantilla, resolver_plantilla
from fantasy_copilot.puntos import ErrorPuntos, obtener_puntos
from fantasy_copilot.textos import preparar_textos
from fantasy_copilot.transcripciones import obtener_transcripciones, separar_copias
from fantasy_copilot.videos import ErrorYouTube, obtener_videos

JORNADAS = range(1, 39)
AVISO = (
    "Aviso: los datos proceden de YouTube (transcripciones de vídeos de analistas) y de "
    "Transfermarkt (lista oficial de jugadores).\n"
    "Se usan con fines de uso educativo y no comercial; los derechos pertenecen a sus autores."
)
AVISO_EVALUACION = (
    "Aviso: los puntos de LaLiga Fantasy proceden de futbolfantasy.com y la lista oficial de "
    "jugadores, de Transfermarkt.\n"
    "Se usan con fines de uso educativo y no comercial; los derechos pertenecen a sus autores."
)
ERRORES_PREVISTOS = (
    ErrorConfiguracion,
    ErrorTransfermarkt,
    ErrorYouTube,
    ErrorInsightsGuardados,
    ErrorDecisionesGuardadas,
    ErrorInsightNoNormalizado,
    ErrorPlantilla,
    ErrorRecomendacion,
    ErrorConjunto,
    OpenAIError,
)


class _ErrorComando(Exception):
    pass


@dataclass
class Servicios:
    youtube: Callable[[str], Any]
    transcripciones: Any
    transfermarkt: Any
    openai: Callable[[str], Any]
    esperar: Callable[[float], None]
    futbolfantasy: Any = None
    juez: Callable[[str, str], Any] | None = None
    ahora: Callable[[], datetime] | None = None


def _servicios_reales() -> Servicios:
    import requests
    from googleapiclient.discovery import build
    from openai import AsyncOpenAI, OpenAI
    from youtube_transcript_api import YouTubeTranscriptApi

    def juez(clave: str, modelo: str):
        from fantasy_copilot.evaluacion import crear_juez

        return crear_juez(modelo, AsyncOpenAI(api_key=clave))

    return Servicios(
        youtube=lambda clave: build("youtube", "v3", developerKey=clave, cache_discovery=False),
        transcripciones=YouTubeTranscriptApi,
        transfermarkt=requests,
        openai=lambda clave: OpenAI(api_key=clave),
        esperar=time.sleep,
        futbolfantasy=requests,
        juez=juez,
        ahora=lambda: datetime.now().astimezone(),
    )


def main(argv: list[str] | None = None, servicios: Servicios | None = None) -> int:
    parser = _parser()
    argumentos = parser.parse_args(argv)
    if argumentos.comando == "recomendar" and not argumentos.plantilla.is_file():
        parser.error(f"--plantilla: no existe el fichero {argumentos.plantilla}")

    servicios = servicios or _servicios_reales()
    try:
        if argumentos.comando == "ingesta":
            _ingesta(argumentos.jornada, argumentos.temporada, servicios)
        elif argumentos.comando == "evaluar":
            _evaluar(servicios)
        else:
            _recomendar(argumentos.jornada, argumentos.temporada, argumentos.plantilla, servicios)
    except (_ErrorComando, *ERRORES_PREVISTOS) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fantasy-copilot",
        description="Recomienda alineaciones de LaLiga Fantasy a partir de vídeos de YouTube.",
    )
    subcomandos = parser.add_subparsers(dest="comando", required=True)
    ingesta = subcomandos.add_parser("ingesta", help="Prepara los datos de una jornada")
    recomendar = subcomandos.add_parser("recomendar", help="Recomienda el once de una plantilla")
    for subcomando in (ingesta, recomendar):
        subcomando.add_argument("--jornada", type=_jornada, required=True)
        subcomando.add_argument(
            "--temporada", type=int, required=True, help="Año en que empieza la temporada"
        )
    recomendar.add_argument("--plantilla", type=Path, required=True)
    subcomandos.add_parser(
        "evaluar", help="Evalúa el pipeline con los conjuntos etiquetados y escribe un informe"
    )
    return parser


def _jornada(valor: str) -> int:
    try:
        jornada = int(valor)
    except ValueError:
        jornada = None
    if jornada not in JORNADAS:
        raise argparse.ArgumentTypeError(
            f"la jornada debe ser un número de {JORNADAS.start} a {JORNADAS.stop - 1}: {valor!r}"
        )
    return jornada


def _ingesta(jornada: int, temporada: int, servicios: Servicios) -> None:
    print(AVISO)
    configuracion = cargar_configuracion()
    openai = servicios.openai(configuracion.clave_openai)
    youtube = servicios.youtube(configuracion.clave_youtube)
    datos_dir = configuracion.datos_dir

    jugadores = obtener_jugadores(
        temporada, datos_dir=datos_dir, cliente=servicios.transfermarkt, esperar=servicios.esperar
    )
    videos = obtener_videos(jornada, temporada, cliente=youtube)
    if not videos:
        raise _ErrorComando(f"No hay vídeos de la jornada {jornada} de la temporada {temporada}")
    transcripciones = obtener_transcripciones(
        videos,
        datos_dir=datos_dir,
        proxy=configuracion.proxy,
        crear_cliente=servicios.transcripciones,
    )
    sin_copias, copias = separar_copias(transcripciones)
    extraccion = extraer_insights(
        preparar_textos(sin_copias),
        jornada=jornada,
        temporada=temporada,
        datos_dir=datos_dir,
        modelo=configuracion.modelo_extraccion,
        cliente=openai,
        esperar=servicios.esperar,
    )
    normalizacion = normalizar_insights(
        extraccion["insights"],
        jugadores,
        jornada=jornada,
        temporada=temporada,
        datos_dir=datos_dir,
        modelo=configuracion.modelo_normalizacion,
        cliente=openai,
    )

    obtenidas = sum(1 for transcripcion in transcripciones if transcripcion["estado"] == "ok")
    print(
        f"\nResumen de la jornada {jornada} de la temporada {temporada}\n"
        f"Vídeos encontrados: {len(videos)}\n"
        f"Transcripciones obtenidas: {obtenidas}\n"
        f"Transcripciones no disponibles: {len(transcripciones) - obtenidas}\n"
        f"Transcripciones duplicadas: {len(copias)}\n"
        f"Insights extraídos: {len(extraccion['insights'])}\n"
        f"Textos fallidos: {len(extraccion['fallidos'])}\n"
        f"Insights normalizados: {len(normalizacion['insights'])}\n"
        f"Nombres eliminados: {len(normalizacion['eliminados'])}"
    )


def _recomendar(jornada: int, temporada: int, ruta_plantilla: Path, servicios: Servicios) -> None:
    configuracion = cargar_configuracion()
    datos_dir = configuracion.datos_dir
    fichero = Path(datos_dir) / str(temporada) / f"j{jornada:02d}" / "insights_normalizados.json"
    if not fichero.exists():
        raise _ErrorComando(
            f"No hay datos de la jornada {jornada} de la temporada {temporada}: ejecuta antes "
            f"fantasy-copilot ingesta --jornada {jornada} --temporada {temporada}"
        )
    openai = servicios.openai(configuracion.clave_openai)
    jugadores = obtener_jugadores(
        temporada, datos_dir=datos_dir, cliente=servicios.transfermarkt, esperar=servicios.esperar
    )
    plantilla = cargar_plantilla(ruta_plantilla, jugadores)

    indice = Indice(cliente=openai, modelo=configuracion.modelo_embeddings)
    indice.indexar(json.loads(fichero.read_text(encoding="utf-8")))
    recomendacion = recomendar_once(
        plantilla,
        jornada=jornada,
        temporada=temporada,
        indice=indice,
        modelo=configuracion.modelo_agente,
        cliente=openai,
        esperar=servicios.esperar,
    )["recomendacion"]

    print(f"Once recomendado para la jornada {jornada} de la temporada {temporada}\n")
    for jugador in recomendacion["once_recomendado"]:
        print(
            f"- {jugador['jugador']} ({jugador['posicion']}) · confianza "
            f"{jugador['confianza']:.2f} · {jugador['justificacion']}"
        )
    if recomendacion["advertencias"]:
        print("\nAdvertencias:")
        for advertencia in recomendacion["advertencias"]:
            print(f"- {advertencia}")
    print(f"\nResumen: {recomendacion['resumen']}")


def _evaluar(servicios: Servicios) -> None:
    print(AVISO_EVALUACION)
    configuracion = cargar_configuracion()
    datos_dir = Path(configuracion.datos_dir)
    ficheros = {conjunto: fichero_conjunto(datos_dir, conjunto) for conjunto in CONJUNTOS}
    if not any(fichero.exists() for fichero in ficheros.values()):
        raise _ErrorComando(
            "No hay ningún conjunto etiquetado; se esperaban "
            + ", ".join(str(fichero) for fichero in ficheros.values())
        )
    # Se validan todos los conjuntos antes de pedir nada a OpenAI.
    conjuntos = {
        conjunto: leer_conjunto(datos_dir, conjunto)
        for conjunto, fichero in ficheros.items()
        if fichero.exists()
    }

    def jugadores_de(temporada: int) -> list[dict]:
        return obtener_jugadores(
            temporada,
            datos_dir=datos_dir,
            cliente=servicios.transfermarkt,
            esperar=servicios.esperar,
        )

    casos = None
    if "recomendacion" in conjuntos:
        casos = []
        for posicion, caso in enumerate(conjuntos["recomendacion"], start=1):
            jugadores = jugadores_de(caso["temporada"])
            try:
                plantilla = resolver_plantilla(
                    caso["plantilla"], jugadores, origen=f"{ficheros['recomendacion']}"
                )
            except ErrorPlantilla as error:
                raise ErrorConjunto(
                    f"La plantilla del caso {posicion} del conjunto "
                    f"{ficheros['recomendacion']} no vale: {error}"
                ) from error
            casos.append({**caso, "plantilla": plantilla, "jugadores": jugadores})

    openai = servicios.openai(configuracion.clave_openai)
    juez = None
    if "extraccion" in conjuntos or casos:
        try:
            juez = servicios.juez(configuracion.clave_openai, configuracion.modelo_agente)
        except ImportError as error:
            raise _ErrorComando(
                f'Evaluar necesita el extra eval: pip install -e ".[eval]" ({error})'
            ) from error

    nombres = None
    if "nombres" in conjuntos:
        nombres = evaluar_nombres(
            datos_dir=datos_dir,
            jugadores_de=jugadores_de,
            modelo=configuracion.modelo_normalizacion,
            cliente=openai,
        )
    extraccion = fidelidad_insights = None
    if "extraccion" in conjuntos:
        extraccion = evaluar_extraccion(
            datos_dir=datos_dir,
            modelo=configuracion.modelo_extraccion,
            cliente=openai,
            esperar=servicios.esperar,
        )
        fidelidad_insights = evaluar_fidelidad_insights(extraccion, juez=juez)
    evaluados = None
    if casos is not None:
        evaluados = [
            _evaluar_caso(caso, datos_dir, configuracion, openai, juez, servicios) for caso in casos
        ]

    resultado = {
        "fecha": servicios.ahora().date().isoformat(),
        "modelos": {
            "extraccion": configuracion.modelo_extraccion,
            "normalizacion": configuracion.modelo_normalizacion,
            "agente": configuracion.modelo_agente,
            "embeddings": configuracion.modelo_embeddings,
        },
        "ficheros": ficheros,
        "nombres": nombres,
        "extraccion": extraccion,
        "fidelidad_insights": fidelidad_insights,
        "casos": evaluados,
    }
    ruta = datos_dir / "evaluacion" / "informes" / f"{resultado['fecha']}.md"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(informe.redactar(resultado), encoding="utf-8")
    print(informe.resumen(resultado, ruta))


def _evaluar_caso(caso, datos_dir, configuracion, openai, juez, servicios) -> dict:
    jornada, temporada, plantilla = caso["jornada"], caso["temporada"], caso["plantilla"]
    evaluado = {"jornada": jornada, "temporada": temporada}

    def saltar(motivo: str) -> dict:
        aviso = f"se salta el caso de la jornada {jornada} de la temporada {temporada}: {motivo}"
        print(f"Aviso: {aviso}", file=sys.stderr)
        return {**evaluado, "aviso": motivo}

    indice = Indice(cliente=openai, modelo=configuracion.modelo_embeddings)
    try:
        recuperacion = evaluar_recuperacion(
            plantilla, jornada=jornada, temporada=temporada, datos_dir=datos_dir, indice=indice
        )
    except ErrorJornadaSinIngerir as error:
        return saltar(str(error))

    ahora = servicios.ahora()
    for previa in range(1, jornada + 1):
        try:
            puntos = obtener_puntos(
                previa,
                temporada,
                jugadores=caso["jugadores"],
                datos_dir=datos_dir,
                cliente=servicios.futbolfantasy,
                ahora=ahora,
            )
        except ErrorPuntos as error:
            if previa == jornada:
                return saltar(str(error))
            print(f"Aviso: {error}", file=sys.stderr)
            continue
        if previa == jornada and puntos["provisional"]:
            return saltar("sus puntos aún son provisionales")

    try:
        recomendacion = recomendar_once(
            plantilla,
            jornada=jornada,
            temporada=temporada,
            indice=indice,
            modelo=configuracion.modelo_agente,
            cliente=openai,
            esperar=servicios.esperar,
        )
        once = evaluar_once(
            plantilla,
            jornada=jornada,
            temporada=temporada,
            datos_dir=datos_dir,
            indice=indice,
            modelo=configuracion.modelo_agente,
            cliente=openai,
            recomendacion=recomendacion,
        )
    except (ErrorRecomendacion, ErrorPuntos, ErrorPlantilla) as error:
        return saltar(str(error))
    return {
        **evaluado,
        "recuperacion": recuperacion,
        "once": once,
        "fidelidad": evaluar_fidelidad(recomendacion, juez=juez),
    }
