import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import openai
import pytest
from youtube_transcript_api import TranscriptsDisabled

from fantasy_copilot.comandos import Servicios, main
from fantasy_copilot.jugadores import obtener_jugadores
from fantasy_copilot.plantilla import ErrorPlantilla, cargar_plantilla

JORNADA = 12
TEMPORADA = 2026

VARIABLES = (
    "OPENAI_API_KEY",
    "YOUTUBE_API_KEY",
    "WEBSHARE_PROXY_USERNAME",
    "WEBSHARE_PROXY_PASSWORD",
    "FANTASY_DATOS_DIR",
    "FANTASY_MODELO_EXTRACCION",
    "FANTASY_MODELO_NORMALIZACION",
    "FANTASY_MODELO_AGENTE",
    "FANTASY_MODELO_EMBEDDINGS",
)


@pytest.fixture(autouse=True)
def datos(tmp_path, monkeypatch):
    """Entorno aislado con claves de prueba y la carpeta de datos dentro de `tmp_path`."""
    for variable in VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-prueba")
    monkeypatch.setenv("YOUTUBE_API_KEY", "yt-prueba")
    monkeypatch.setenv("FANTASY_DATOS_DIR", str(tmp_path / "datos"))
    return tmp_path / "datos"


# --- Transfermarkt: una sola página de LaLiga con un equipo de doce jugadores ---------------

OFICIALES = [
    ("121483", "Jan Oblak", "Portero"),
    ("411975", "Jules Koundé", "Lateral derecho"),
    ("937945", "Pau Cubarsí", "Defensa central"),
    ("462250", "Dani Vivian", "Defensa central"),
    ("683723", "Miguel Gutiérrez", "Lateral izquierdo"),
    ("683840", "Pedri", "Mediocentro"),
    ("369081", "Federico Valverde", "Mediocentro"),
    ("937958", "Lamine Yamal", "Extremo derecho"),
    ("411295", "Raphinha", "Extremo izquierdo"),
    ("576024", "Julián Álvarez", "Delantero centro"),
    ("709187", "Nico Williams", "Extremo izquierdo"),
    ("264335", "Augusto Batalla", "Portero"),
]

BASE = "https://www.transfermarkt.es"
URL_LIGA = f"{BASE}/laliga/startseite/wettbewerb/ES1/plus/?saison_id={TEMPORADA}"
URL_EQUIPO = f"{BASE}/fc-barcelona/kader/verein/131/saison_id/{TEMPORADA}"
PAGINA_LIGA = f"""
<table class="items"><tbody>
  <tr><td class="hauptlink"><a href="/fc-barcelona/startseite/verein/131/saison_id/{TEMPORADA}">FC Barcelona</a></td></tr>
</tbody></table>
"""  # noqa: E501


def _fila(identificador, nombre, posicion):
    return f"""
  <tr>
    <td class="posrela">
      <table class="inline-table">
        <tr><td rowspan="2"><img src="x.png"></td><td class="hauptlink"><a href="/jugador/profil/spieler/{identificador}">{nombre}</a></td></tr>
        <tr><td>{posicion}</td></tr>
      </table>
    </td>
  </tr>"""  # noqa: E501


PAGINA_EQUIPO = (
    '<table class="items"><tbody>'
    + "".join(_fila(*oficial) for oficial in OFICIALES)
    + "</tbody></table>"
)


@dataclass
class Respuesta:
    status_code: int
    text: str


class TransfermarktFalso:
    """Hace de cliente HTTP de Transfermarkt. Una URL sin página responde 404."""

    def __init__(self, paginas=None):
        if paginas is None:
            paginas = {URL_LIGA: PAGINA_LIGA, URL_EQUIPO: PAGINA_EQUIPO}
        self.paginas = paginas

    def get(self, url, headers=None, **_):
        pagina = self.paginas.get(url)
        return Respuesta(404, "") if pagina is None else Respuesta(200, pagina)


# --- YouTube Data API: los vídeos se publican en el canal de carrasco --------------------------


def video(video_id, titulo):
    return {
        "snippet": {
            "title": titulo,
            "publishedAt": "2026-11-05T10:00:00Z",
            "resourceId": {"videoId": video_id},
        }
    }


VIDEO_UNO = video("videouno01", "J12 onces probables")
VIDEO_DOS = video("videodos002", "Jornada 12: chollos y descartes")
CARRASCO = "UCu7ZOzCUgWpvvJzYAdnw0_Q"


class _Peticion:
    def __init__(self, respuesta):
        self._respuesta = respuesta

    def execute(self):
        return self._respuesta


class _Recurso:
    def __init__(self, responder):
        self._responder = responder

    def list(self, **parametros):
        return _Peticion(self._responder(**parametros))


class YouTubeFalso:
    """Hace de cliente de YouTube Data API: una sola página de subidas por canal."""

    def __init__(self, videos=(VIDEO_UNO, VIDEO_DOS)):
        self.videos = list(videos)

    def channels(self):
        return _Recurso(self._canal)

    def playlistItems(self):
        return _Recurso(self._pagina)

    def _canal(self, id, **_):
        return {"items": [{"contentDetails": {"relatedPlaylists": {"uploads": "UU" + id[2:]}}}]}

    def _pagina(self, playlistId, **_):
        return {"items": self.videos if playlistId == "UU" + CARRASCO[2:] else []}


# --- youtube-transcript-api -------------------------------------------------------------------


@dataclass
class Fragmento:
    text: str
    start: float
    duration: float


def transcripcion(marca):
    relleno = "el analista repasa con calma las alineaciones de la jornada " * 12
    return [Fragmento(f"{marca} {relleno}", 0.0, 60.0)]


class TranscripcionesFalsas:
    """Hace de `YouTubeTranscriptApi`. Un vídeo sin entrada no tiene transcripción."""

    def __init__(self, respuestas=None):
        self.respuestas = (
            {"videouno01": transcripcion("marcadoruno")} if respuestas is None else respuestas
        )

    def __call__(self, proxy_config=None, **_):
        return self

    def fetch(self, video_id, languages=("en",), **_):
        if video_id not in self.respuestas:
            raise TranscriptsDisabled(video_id)
        return self.respuestas[video_id]


# --- OpenAI -----------------------------------------------------------------------------------

PLANTILLA = [
    {"nombre": "Jan Oblak", "posicion": "portero"},
    {"nombre": "Jules Koundé", "posicion": "defensa"},
    {"nombre": "Pau Cubarsí", "posicion": "defensa"},
    {"nombre": "Dani Vivian", "posicion": "defensa"},
    {"nombre": "Miguel Gutiérrez", "posicion": "defensa"},
    {"nombre": "Pedri", "posicion": "centrocampista"},
    {"nombre": "Federico Valverde", "posicion": "centrocampista"},
    {"nombre": "Lamine Yamal", "posicion": "centrocampista"},
    {"nombre": "Raphinha", "posicion": "delantero"},
    {"nombre": "Julián Álvarez", "posicion": "delantero"},
    {"nombre": "Nico Williams", "posicion": "delantero"},
    {"nombre": "Augusto Batalla", "posicion": "portero"},
]
NOMBRES = [jugador["nombre"] for jugador in PLANTILLA]
POSICION = {jugador["nombre"]: jugador["posicion"] for jugador in PLANTILLA}
ONCE = PLANTILLA[:11]
JUSTIFICACION = {
    jugador["nombre"]: f"Justificación {letra}"
    for jugador, letra in zip(ONCE, "ABCDEFGHIJK", strict=True)
}
ADVERTENCIA = "Vigilar las rotaciones entre semana"
RESUMEN_FINAL = "Once ofensivo aprovechando el calendario"
RAZON_RAFINA = "Marca en cada partido"

INSIGHT_RAFINA = {
    "jugador": "Rafiña",
    "accion": "alinear",
    "confianza": 0.9,
    "razon_corta": RAZON_RAFINA,
    "criterios": {
        "forma_actual": None,
        "dinamica_equipo": None,
        "calendario_rival": None,
        "prediccion_puntos": None,
    },
    "quote_original": "Rafiña es fijo esta jornada",
}
INSIGHT_INVENTADO = {**INSIGHT_RAFINA, "jugador": "Zzyzx Qwerty", "quote_original": None}

_ESCAPE_UNICODE = re.compile(r"\\+u([0-9a-fA-F]{4})")


def _respuesta(contenido):
    mensaje = SimpleNamespace(
        role="assistant",
        content=json.dumps(contenido, ensure_ascii=False),
        refusal=None,
        tool_calls=None,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=mensaje, finish_reason="stop")])


class OpenAIFalso:
    """Hace de cliente de OpenAI para todas las etapas.

    Cada petición de chat se atiende según el nombre del esquema de respuesta que pide:
    extracción (por la marca de la transcripción), resolución de nombres, resumen por jugador y
    recomendación final. Solo se miran los mensajes que no son instrucciones de sistema.
    `extraccion` asocia una marca con sus insights o con una excepción. Con `fallar`, toda
    petición lanza un error de la API.
    """

    def __init__(self, extraccion=None, fallar=False):
        if extraccion is None:
            extraccion = {"marcadoruno": [INSIGHT_RAFINA, INSIGHT_INVENTADO]}
        self.extraccion = extraccion
        self.fallar = fallar
        self.resumenes = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._crear))
        self.embeddings = SimpleNamespace(create=self._embeddings)

    def _embeddings(self, *, model, input, **_):
        if self.fallar:
            raise openai.OpenAIError("fallo de conexión")
        textos = [input] if isinstance(input, str) else list(input)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0, 0.0]) for _ in textos])

    def _crear(self, **peticion):
        if self.fallar:
            raise openai.OpenAIError("fallo de conexión")
        sin_instrucciones = [
            mensaje
            for mensaje in peticion.get("messages", [])
            if not (isinstance(mensaje, dict) and mensaje.get("role") == "system")
        ]
        enviado = json.dumps(sin_instrucciones, ensure_ascii=False, default=str)
        enviado = _ESCAPE_UNICODE.sub(lambda escape: chr(int(escape.group(1), 16)), enviado)
        esquema = (peticion.get("response_format") or {}).get("json_schema", {}).get("name")
        if esquema == "insights_extraidos":
            return self._extraer(enviado)
        if esquema == "jugador_resuelto":
            return self._resolver(enviado)
        if esquema == "resumen_jugador":
            return self._resumir(enviado)
        if esquema == "recomendacion_once":
            return _respuesta(
                {
                    "once_recomendado": [
                        {
                            "jugador": jugador["nombre"],
                            "posicion": jugador["posicion"],
                            "confianza": 0.75,
                            "justificacion": JUSTIFICACION[jugador["nombre"]],
                        }
                        for jugador in ONCE
                    ],
                    "advertencias": [ADVERTENCIA],
                    "resumen": RESUMEN_FINAL,
                }
            )
        raise AssertionError(f"Petición inesperada con esquema {esquema!r}")

    def _extraer(self, enviado):
        marca = next((m for m in self.extraccion if m in enviado), None)
        if marca is None:
            raise AssertionError("Petición de extracción sin marca conocida")
        insights = self.extraccion[marca]
        if isinstance(insights, Exception):
            raise insights
        return _respuesta({"insights": insights})

    def _resolver(self, enviado):
        if "Rafiña" in enviado:
            return _respuesta({"jugador_id": "411295"})
        return _respuesta({"jugador_id": None})

    def _resumir(self, enviado):
        nombrados = [nombre for nombre in NOMBRES if nombre in enviado]
        otros = [nombre for nombre in nombrados if nombre != "Raphinha"]
        sujeto = otros[0] if len(otros) == 1 else "Raphinha"
        self.resumenes.append({"jugador": sujeto, "enviado": enviado})
        return _respuesta(
            {
                "jugador": sujeto,
                "posicion": POSICION[sujeto],
                "accion_recomendada": "alinear",
                "confianza_media": 0.8,
                "resumen": "Titular habitual.",
                "fuentes": ["carrasco - Jornada 12"],
            }
        )


# --- Ejecución ---------------------------------------------------------------------------------


def servicios(youtube=None, transcripciones=None, web=None, cliente_openai=None):
    youtube = youtube or YouTubeFalso()
    cliente_openai = cliente_openai or OpenAIFalso()
    return Servicios(
        youtube=lambda clave: youtube,
        transcripciones=transcripciones or TranscripcionesFalsas(),
        transfermarkt=web or TransfermarktFalso(),
        openai=lambda clave: cliente_openai,
        esperar=lambda segundos: None,
    )


@dataclass
class Ejecucion:
    codigo: int
    salida: str
    errores: str


def ejecutar(capsys, *argumentos, con=None):
    """Lanza el comando en el mismo proceso. Una excepción no capturada hace fallar el test."""
    errores_de_salida = ""
    try:
        codigo = main(list(argumentos), servicios=con or servicios())
    except SystemExit as salida:
        codigo = salida.code
        if isinstance(codigo, str):
            errores_de_salida, codigo = codigo, 1
        codigo = codigo or 0
    capturado = capsys.readouterr()
    return Ejecucion(codigo, capturado.out, capturado.err + errores_de_salida)


def ingesta(capsys, jornada=JORNADA, con=None):
    return ejecutar(
        capsys, "ingesta", "--jornada", str(jornada), "--temporada", str(TEMPORADA), con=con
    )


def recomendar(capsys, plantilla, jornada=JORNADA, con=None):
    return ejecutar(
        capsys,
        "recomendar",
        "--jornada",
        str(jornada),
        "--temporada",
        str(TEMPORADA),
        "--plantilla",
        str(plantilla),
        con=con,
    )


def escribir_plantilla(carpeta, plantilla=PLANTILLA):
    ruta = carpeta / "plantilla.json"
    ruta.write_text(json.dumps(plantilla, ensure_ascii=False), encoding="utf-8")
    return ruta


def normalizados(datos, jornada=JORNADA):
    return datos / str(TEMPORADA) / f"j{jornada}" / "insights_normalizados.json"


def cifra(salida, etiqueta):
    """La cifra que acompaña a `etiqueta` en las últimas líneas de la salida.

    Admite `2 vídeos encontrados`, `1 transcripción obtenida` o `Vídeos encontrados: 2`.
    """
    final = "\n".join([linea for linea in salida.splitlines() if linea.strip()][-10:])
    espacio = r"[^\S\n]"
    patron = (
        rf"(\d+){espacio}+(?:\w+{espacio}+)?{etiqueta}"
        rf"|{etiqueta}\w*(?:{espacio}+\w+)?{espacio}*:{espacio}*(\d+)"
    )
    encontrada = re.search(patron, final, re.IGNORECASE)
    assert encontrada, f"El resumen final no da la cifra de {etiqueta!r}:\n{final}"
    return int(encontrada.group(1) or encontrada.group(2))


# --- Tests -------------------------------------------------------------------------------------


@pytest.mark.parametrize("jornada", [0, 39])
def test_una_jornada_fuera_de_1_a_38_termina_con_error_que_nombra_el_argumento(capsys, jornada):
    resultado = ingesta(capsys, jornada=jornada)

    assert resultado.codigo != 0
    assert "--jornada" in resultado.errores


def test_sin_clave_de_openai_la_ingesta_termina_con_error_que_nombra_la_variable(
    capsys, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY")

    resultado = ingesta(capsys)

    assert resultado.codigo != 0
    assert "OPENAI_API_KEY" in resultado.errores


def test_la_ingesta_empieza_con_el_aviso_de_origen_y_uso_de_los_datos(capsys):
    resultado = ingesta(capsys)

    inicio = "\n".join([linea for linea in resultado.salida.splitlines() if linea.strip()][:5])
    assert "YouTube" in inicio
    assert "Transfermarkt" in inicio
    assert "uso educativo y no comercial" in inicio


def test_la_ingesta_deja_los_insights_resueltos_contra_la_lista_oficial_de_transfermarkt(
    capsys, datos
):
    resultado = ingesta(capsys)

    assert resultado.codigo == 0
    guardados = json.loads(normalizados(datos).read_text(encoding="utf-8"))
    resueltos = [(i["nombre_transcrito"], i["jugador_id"]) for i in guardados]
    assert resueltos == [("Rafiña", "411295")]


def test_el_resumen_de_la_ingesta_cuenta_videos_y_transcripciones_obtenidas_y_no_disponibles(
    capsys,
):
    resultado = ingesta(capsys)

    cifras = [cifra(resultado.salida, e) for e in ("vídeo", "obtenida", "no disponible")]
    assert cifras == [2, 1, 1]


def test_el_resumen_de_la_ingesta_cuenta_insights_textos_fallidos_normalizados_y_eliminados(
    capsys,
):
    transcripciones = TranscripcionesFalsas(
        {
            "videouno01": transcripcion("marcadoruno"),
            "videodos002": transcripcion("marcadordos"),
        }
    )
    cliente = OpenAIFalso(
        extraccion={
            "marcadoruno": [INSIGHT_RAFINA, INSIGHT_INVENTADO],
            "marcadordos": openai.OpenAIError("fallo de conexión"),
        }
    )

    resultado = ingesta(
        capsys, con=servicios(transcripciones=transcripciones, cliente_openai=cliente)
    )

    etiquetas = ("extraído", "fallido", "normalizado", "eliminado")
    assert [cifra(resultado.salida, e) for e in etiquetas] == [2, 1, 1, 1]


def test_sin_videos_de_la_jornada_la_ingesta_lo_dice_y_no_escribe_insights_normalizados(
    capsys, datos
):
    youtube = YouTubeFalso([video("otrajornada", "J11 onces probables")])

    resultado = ingesta(capsys, con=servicios(youtube=youtube))

    assert resultado.codigo != 0
    assert "vídeo" in resultado.errores.lower()
    assert not normalizados(datos).exists()


def test_recomendar_sin_datos_de_la_jornada_pide_ejecutar_antes_la_ingesta_de_esa_jornada(
    capsys, tmp_path
):
    ingesta(capsys)

    resultado = recomendar(capsys, escribir_plantilla(tmp_path), jornada=13)

    assert resultado.codigo != 0
    assert "ingesta" in resultado.errores
    assert "13" in resultado.errores
    assert str(TEMPORADA) in resultado.errores


def test_una_plantilla_que_no_existe_termina_con_error_que_nombra_el_argumento(capsys, tmp_path):
    ingesta(capsys)

    resultado = recomendar(capsys, tmp_path / "no_existe.json")

    assert resultado.codigo != 0
    assert "--plantilla" in resultado.errores


def test_una_plantilla_no_valida_termina_con_el_mensaje_de_su_validacion(capsys, tmp_path, datos):
    ingesta(capsys)
    ruta = escribir_plantilla(tmp_path, PLANTILLA[:10])
    with pytest.raises(ErrorPlantilla) as error:
        oficiales = obtener_jugadores(
            TEMPORADA, datos_dir=datos, cliente=TransfermarktFalso(), esperar=lambda s: None
        )
        cargar_plantilla(ruta, oficiales)

    resultado = recomendar(capsys, ruta)

    assert resultado.codigo != 0
    assert str(error.value) in resultado.errores


def test_recomendar_imprime_el_once_seguido_de_advertencias_y_resumen(capsys, tmp_path):
    ingesta(capsys)

    resultado = recomendar(capsys, escribir_plantilla(tmp_path))

    assert resultado.codigo == 0
    lineas = resultado.salida.splitlines()
    posiciones_del_once = []
    for jugador in ONCE:
        indice = next(
            (n for n, linea in enumerate(lineas) if JUSTIFICACION[jugador["nombre"]] in linea),
            None,
        )
        assert indice is not None, f"Falta la justificación de {jugador['nombre']}"
        linea = lineas[indice]
        assert jugador["nombre"] in linea
        assert jugador["posicion"] in linea
        assert re.search(r"0[.,]75|75\s?%", linea), f"Falta la confianza en: {linea}"
        posiciones_del_once.append(indice)
    advertencia = next(n for n, linea in enumerate(lineas) if ADVERTENCIA in linea)
    resumen = next(n for n, linea in enumerate(lineas) if RESUMEN_FINAL in linea)
    assert max(posiciones_del_once) < advertencia < resumen


def test_la_recomendacion_resume_a_cada_jugador_con_sus_insights_normalizados(capsys, tmp_path):
    cliente = OpenAIFalso()
    ingesta(capsys, con=servicios(cliente_openai=cliente))

    recomendar(capsys, escribir_plantilla(tmp_path), con=servicios(cliente_openai=cliente))

    de_raphinha = [r["enviado"] for r in cliente.resumenes if r["jugador"] == "Raphinha"]
    assert any(RAZON_RAFINA in enviado for enviado in de_raphinha)


def test_una_segunda_ingesta_aprovecha_lo_guardado_sin_volver_a_consultar_los_servicios(
    capsys, datos
):
    ingesta(capsys)
    primera = normalizados(datos).read_text(encoding="utf-8")
    sin_servicio = servicios(
        web=TransfermarktFalso(paginas={}),
        cliente_openai=OpenAIFalso(fallar=True),
    )

    resultado = ingesta(capsys, con=sin_servicio)

    assert resultado.codigo == 0
    assert normalizados(datos).read_text(encoding="utf-8") == primera


def test_el_paquete_instala_el_comando_fantasy_copilot(tmp_path):
    comando = Path(sys.executable).parent / "fantasy-copilot"
    entorno = {k: v for k, v in os.environ.items() if k not in VARIABLES}
    assert comando.exists(), "El paquete no instala el comando fantasy-copilot"

    resultado = subprocess.run(
        [str(comando), "ingesta", "--jornada", "0", "--temporada", str(TEMPORADA)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=entorno,
    )

    assert resultado.returncode != 0
    assert "--jornada" in resultado.stderr
    assert "Traceback" not in resultado.stderr
