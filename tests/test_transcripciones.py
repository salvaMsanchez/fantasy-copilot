import json
from dataclasses import dataclass

import pytest
import requests
from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled
from youtube_transcript_api.proxies import WebshareProxyConfig

from fantasy_copilot.configuracion import cargar_configuracion
from fantasy_copilot.transcripciones import obtener_transcripciones


@dataclass
class Fragmento:
    text: str
    start: float
    duration: float


FRAGMENTOS = [
    Fragmento("hola a todos", 0.0, 2.5),
    Fragmento("vamos con la jornada", 2.5, 3.0),
]

TRANSCRIPCION_OK = {
    "estado": "ok",
    "motivo": None,
    "transcripcion_cruda": "hola a todos vamos con la jornada",
    "fragmentos": [
        {"texto": "hola a todos", "inicio": 0.0, "duracion": 2.5},
        {"texto": "vamos con la jornada", "inicio": 2.5, "duracion": 3.0},
    ],
}


def video(video_id):
    return {
        "canal": "carrasco",
        "video_id": video_id,
        "titulo": "Jornada 12",
        "fecha": "2026-11-05",
        "jornada": 12,
        "temporada": 2026,
        "url": f"https://youtube.com/watch?v={video_id}",
    }


class YouTubeFalso:
    """Hace de `YouTubeTranscriptApi`: se construye con `proxy_config` y responde a `fetch`.

    `respuestas` asocia cada vídeo con sus fragmentos en español o con la excepción que lanza.
    """

    def __init__(self, respuestas):
        self.respuestas = respuestas
        self.proxies = []
        self.consultados = []

    def __call__(self, proxy_config=None, **_):
        self.proxies.append(proxy_config)
        return self

    def fetch(self, video_id, languages=("en",), **_):
        self.consultados.append(video_id)
        respuesta = self.respuestas[video_id]
        if isinstance(respuesta, Exception):
            raise respuesta
        if "es" not in languages:
            raise NoTranscriptFound(video_id, languages, "solo hay subtítulos en español")
        return respuesta


class YouTubeSoloEnIngles:
    def __call__(self, proxy_config=None, **_):
        return self

    def fetch(self, video_id, languages=("en",), **_):
        if "en" in languages:
            return [Fragmento("hello everyone", 0.0, 2.5)]
        raise NoTranscriptFound(video_id, languages, "solo hay subtítulos en inglés")


def obtener(videos, datos_dir, youtube, proxy=None):
    return obtener_transcripciones(videos, datos_dir=datos_dir, proxy=proxy, crear_cliente=youtube)


def fichero(datos_dir, video_id):
    return datos_dir / "transcripciones" / f"{video_id}.json"


def test_de_los_fragmentos_en_espanol_se_obtiene_el_texto_unido_y_cada_fragmento(tmp_path):
    youtube = YouTubeFalso({"AT_UTIeNmtI": FRAGMENTOS})

    [transcripcion] = obtener([video("AT_UTIeNmtI")], tmp_path, youtube)

    assert {clave: transcripcion[clave] for clave in TRANSCRIPCION_OK} == TRANSCRIPCION_OK


def test_el_resultado_se_guarda_en_el_fichero_del_video_con_sus_datos(tmp_path):
    youtube = YouTubeFalso({"AT_UTIeNmtI": FRAGMENTOS})

    obtener([video("AT_UTIeNmtI")], tmp_path / "datos", youtube)

    guardado = json.loads(fichero(tmp_path / "datos", "AT_UTIeNmtI").read_text(encoding="utf-8"))
    assert guardado == {**video("AT_UTIeNmtI"), **TRANSCRIPCION_OK}


def test_si_el_fichero_del_video_existe_con_estado_ok_se_devuelve_sin_consultar_youtube(tmp_path):
    guardado = {
        **video("AT_UTIeNmtI"),
        "estado": "ok",
        "motivo": None,
        "transcripcion_cruda": "texto ya guardado",
        "fragmentos": [{"texto": "texto ya guardado", "inicio": 1.0, "duracion": 4.0}],
    }
    fichero(tmp_path, "AT_UTIeNmtI").parent.mkdir(parents=True)
    fichero(tmp_path, "AT_UTIeNmtI").write_text(json.dumps(guardado), encoding="utf-8")
    youtube = YouTubeFalso({"AT_UTIeNmtI": FRAGMENTOS})

    transcripciones = obtener([video("AT_UTIeNmtI")], tmp_path, youtube)

    assert transcripciones == [guardado]
    assert youtube.consultados == []


def test_un_video_sin_subtitulos_en_espanol_queda_no_disponible_sin_probar_otro_idioma(tmp_path):
    [transcripcion] = obtener([video("AT_UTIeNmtI")], tmp_path, YouTubeSoloEnIngles())

    assert isinstance(transcripcion["motivo"], str) and transcripcion["motivo"].strip()
    assert {**transcripcion, "motivo": None} == {
        **video("AT_UTIeNmtI"),
        "estado": "no_disponible",
        "motivo": None,
        "transcripcion_cruda": None,
        "fragmentos": [],
    }


@pytest.mark.parametrize(
    "fallo",
    [TranscriptsDisabled("segundo"), requests.ConnectionError("conexión rechazada")],
    ids=["subtitulos_desactivados", "error_de_red"],
)
def test_si_falla_la_descarga_de_un_video_queda_no_disponible_y_los_demas_se_procesan(
    tmp_path, fallo
):
    youtube = YouTubeFalso({"primero": FRAGMENTOS, "segundo": fallo, "tercero": FRAGMENTOS})

    transcripciones = obtener(
        [video("primero"), video("segundo"), video("tercero")], tmp_path, youtube
    )

    assert [t["estado"] for t in transcripciones] == ["ok", "no_disponible", "ok"]
    assert isinstance(transcripciones[1]["motivo"], str) and transcripciones[1]["motivo"].strip()


def test_un_video_guardado_como_no_disponible_se_vuelve_a_consultar(tmp_path):
    guardado = {
        **video("AT_UTIeNmtI"),
        "estado": "no_disponible",
        "motivo": "subtítulos desactivados",
        "transcripcion_cruda": None,
        "fragmentos": [],
    }
    fichero(tmp_path, "AT_UTIeNmtI").parent.mkdir(parents=True)
    fichero(tmp_path, "AT_UTIeNmtI").write_text(json.dumps(guardado), encoding="utf-8")
    youtube = YouTubeFalso({"AT_UTIeNmtI": FRAGMENTOS})

    transcripciones = obtener([video("AT_UTIeNmtI")], tmp_path, youtube)

    assert transcripciones == [{**video("AT_UTIeNmtI"), **TRANSCRIPCION_OK}]


def test_sin_proxy_configurado_la_descarga_se_hace_sin_proxy(tmp_path):
    youtube = YouTubeFalso({"AT_UTIeNmtI": FRAGMENTOS})

    obtener([video("AT_UTIeNmtI")], tmp_path, youtube, proxy=None)

    assert youtube.consultados == ["AT_UTIeNmtI"]
    assert set(youtube.proxies) == {None}


def test_con_proxy_configurado_la_descarga_usa_webshare_con_sus_credenciales(tmp_path, monkeypatch):
    monkeypatch.setenv("WEBSHARE_PROXY_USERNAME", "usuario")
    monkeypatch.setenv("WEBSHARE_PROXY_PASSWORD", "secreto")
    youtube = YouTubeFalso({"AT_UTIeNmtI": FRAGMENTOS})

    obtener([video("AT_UTIeNmtI")], tmp_path, youtube, proxy=cargar_configuracion().proxy)

    assert youtube.consultados == ["AT_UTIeNmtI"]
    assert {(type(p), p.proxy_username, p.proxy_password) for p in youtube.proxies} == {
        (WebshareProxyConfig, "usuario", "secreto")
    }
