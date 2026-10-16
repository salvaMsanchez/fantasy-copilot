import json
import re
from pathlib import Path

from youtube_transcript_api import NoTranscriptFound, YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig

from fantasy_copilot.configuracion import Proxy

IDIOMA = "es"
LONGITUD_MAXIMA_MOTIVO = 200
PALABRAS_POR_SECUENCIA = 3
PARECIDO_MINIMO_COPIA = 0.5


def separar_copias(transcripciones: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separa las transcripciones obtenidas que repiten el contenido de otra del mismo canal.

    De cada grupo de copias se conserva la del vídeo publicado antes. El parecido es la parte
    de las secuencias de tres palabras de la más corta que aparecen también en la otra.
    """
    obtenidas = [
        (posicion, transcripcion)
        for posicion, transcripcion in enumerate(transcripciones)
        if transcripcion["estado"] == "ok"
    ]
    # La lista llega del vídeo más reciente al más antiguo: a igual fecha, va antes el de detrás.
    obtenidas.sort(key=lambda par: (par[1]["fecha"], -par[0]))

    conservadas: list[tuple[dict, set]] = []
    copias = []
    for _, transcripcion in obtenidas:
        secuencias = _secuencias(transcripcion["transcripcion_cruda"])
        if any(
            otra["canal"] == transcripcion["canal"]
            and _parecido(secuencias, secuencias_otra) >= PARECIDO_MINIMO_COPIA
            for otra, secuencias_otra in conservadas
        ):
            copias.append(transcripcion)
        else:
            conservadas.append((transcripcion, secuencias))

    ids_copias = {id(copia) for copia in copias}
    return [t for t in transcripciones if id(t) not in ids_copias], copias


def _secuencias(texto: str) -> set[tuple[str, ...]]:
    palabras = re.findall(r"\w+", texto.lower())
    n = PALABRAS_POR_SECUENCIA
    return {tuple(palabras[i : i + n]) for i in range(len(palabras) - n + 1)}


def _parecido(unas: set, otras: set) -> float:
    if not unas or not otras:
        return 0.0
    return len(unas & otras) / min(len(unas), len(otras))


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
