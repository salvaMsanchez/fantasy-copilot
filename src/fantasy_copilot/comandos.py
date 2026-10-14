import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAIError

from fantasy_copilot.agente import ErrorRecomendacion, recomendar_once
from fantasy_copilot.configuracion import ErrorConfiguracion, cargar_configuracion
from fantasy_copilot.indice import ErrorInsightNoNormalizado, Indice
from fantasy_copilot.insights import ErrorInsightsGuardados, extraer_insights
from fantasy_copilot.jugadores import ErrorTransfermarkt, obtener_jugadores
from fantasy_copilot.normalizacion import ErrorDecisionesGuardadas, normalizar_insights
from fantasy_copilot.plantilla import ErrorPlantilla, cargar_plantilla
from fantasy_copilot.textos import preparar_textos
from fantasy_copilot.transcripciones import obtener_transcripciones
from fantasy_copilot.videos import ErrorYouTube, obtener_videos

JORNADAS = range(1, 39)
AVISO = (
    "Aviso: los datos proceden de YouTube (transcripciones de vídeos de analistas) y de "
    "Transfermarkt (lista oficial de jugadores).\n"
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


def _servicios_reales() -> Servicios:
    import requests
    from googleapiclient.discovery import build
    from openai import OpenAI
    from youtube_transcript_api import YouTubeTranscriptApi

    return Servicios(
        youtube=lambda clave: build("youtube", "v3", developerKey=clave, cache_discovery=False),
        transcripciones=YouTubeTranscriptApi,
        transfermarkt=requests,
        openai=lambda clave: OpenAI(api_key=clave),
        esperar=time.sleep,
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
    extraccion = extraer_insights(
        preparar_textos(transcripciones),
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
