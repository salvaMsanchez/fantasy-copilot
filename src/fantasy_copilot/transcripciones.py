import json
import re
from pathlib import Path

from youtube_transcript_api import NoTranscriptFound, YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig

from fantasy_copilot.configuracion import Proxy

IDIOMA = "es"
LONGITUD_MAXIMA_MOTIVO = 200


def obtener_transcripciones(
    videos: list[dict],
    *,
    datos_dir: Path,
    proxy: Proxy | None = None,
    crear_cliente=YouTubeTranscriptApi,
) -> list[dict]:
    directorio = Path(datos_dir) / "transcripciones"
    directorio.mkdir(parents=True, exist_ok=True)
    proxy_config = None
    if proxy is not None:
        proxy_config = WebshareProxyConfig(
            proxy_username=proxy.usuario, proxy_password=proxy.contrasena
        )
    cliente = crear_cliente(proxy_config=proxy_config)

    transcripciones = []
    for video in videos:
        fichero = directorio / f"{video['video_id']}.json"
        transcripcion = _leer_descargada(fichero)
        if transcripcion is None:
            transcripcion = {**video, **_descargar(cliente, video["video_id"], proxy)}
            fichero.write_text(
                json.dumps(transcripcion, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        transcripciones.append(transcripcion)
    return transcripciones


def _leer_descargada(fichero: Path) -> dict | None:
    if not fichero.exists():
        return None
    try:
        guardada = json.loads(fichero.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if not isinstance(guardada, dict) or guardada.get("estado") != "ok":
        return None
    return guardada


def _descargar(cliente, video_id: str, proxy: Proxy | None) -> dict:
    try:
        fragmentos = [
            {"texto": fragmento.text, "inicio": fragmento.start, "duracion": fragmento.duration}
            for fragmento in cliente.fetch(video_id, languages=[IDIOMA])
        ]
    except NoTranscriptFound:
        return _no_disponible("El vídeo no tiene subtítulos en español")
    except Exception as error:
        causa = getattr(error, "cause", None) or str(error)
        return _no_disponible(_motivo(f"{type(error).__name__}: {causa}", proxy))
    if not any(fragmento["texto"].strip() for fragmento in fragmentos):
        return _no_disponible("La transcripción no tiene texto")
    return {
        "estado": "ok",
        "motivo": None,
        "transcripcion_cruda": " ".join(fragmento["texto"] for fragmento in fragmentos),
        "fragmentos": fragmentos,
    }


def _motivo(mensaje: str, proxy: Proxy | None) -> str:
    if proxy is not None:
        for secreto in (proxy.usuario, proxy.contrasena):
            if secreto:
                mensaje = mensaje.replace(secreto, "***")
    mensaje = re.sub(r"://[^/\s]+@", "://***@", mensaje)
    mensaje = " ".join(mensaje.split())
    if len(mensaje) > LONGITUD_MAXIMA_MOTIVO:
        mensaje = mensaje[: LONGITUD_MAXIMA_MOTIVO - 1] + "…"
    return mensaje


def _no_disponible(motivo: str) -> dict:
    return {
        "estado": "no_disponible",
        "motivo": motivo,
        "transcripcion_cruda": None,
        "fragmentos": [],
    }
