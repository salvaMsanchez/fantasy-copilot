import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections.faithfulness.util import (
    NLIStatementOutput,
    StatementGeneratorOutput,
)

from fantasy_copilot.comandos import Servicios, main
from fantasy_copilot.evaluacion import ErrorConjunto, evaluar_extraccion

TEMPORADA = 2026
AHORA = datetime(2026, 10, 21, 12, 0, tzinfo=ZoneInfo("Europe/Madrid"))
FECHA = "2026-10-21"

MODELOS = {
    "FANTASY_MODELO_EXTRACCION": "modelo-extraccion-prueba",
    "FANTASY_MODELO_NORMALIZACION": "modelo-normalizacion-prueba",
    "FANTASY_MODELO_AGENTE": "modelo-agente-prueba",
    "FANTASY_MODELO_EMBEDDINGS": "modelo-embeddings-prueba",
}
VARIABLES = (
    "OPENAI_API_KEY",
    "YOUTUBE_API_KEY",
    "WEBSHARE_PROXY_USERNAME",
    "WEBSHARE_PROXY_PASSWORD",
    "FANTASY_DATOS_DIR",
    *MODELOS,
)


@pytest.fixture(autouse=True)
def datos(tmp_path, monkeypatch):
    """Entorno aislado con claves de prueba, modelos reconocibles y los datos en `tmp_path`."""
    for variable in VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-prueba")
    monkeypatch.setenv("FANTASY_DATOS_DIR", str(tmp_path / "datos"))
    for variable, modelo in MODELOS.items():
        monkeypatch.setenv(variable, modelo)
    return tmp_path / "datos"


# --- Transfermarkt: un solo equipo con los jugadores de la plantilla ---------------------------

OFICIALES = [
    ("121483", "Jan Oblak", "Portero", "portero"),
    ("264335", "Augusto Batalla", "Portero", "portero"),
    ("411975", "Jules Koundé", "Lateral derecho", "defensa"),
    ("937945", "Pau Cubarsí", "Defensa central", "defensa"),
    ("462250", "Dani Vivian", "Defensa central", "defensa"),
    ("683723", "Miguel Gutiérrez", "Lateral izquierdo", "defensa"),
    ("636688", "Alejandro Balde", "Lateral izquierdo", "defensa"),
    ("683840", "Pedri", "Mediocentro", "centrocampista"),
    ("369081", "Federico Valverde", "Mediocentro", "centrocampista"),
    ("937958", "Lamine Yamal", "Extremo derecho", "centrocampista"),
    ("411295", "Raphinha", "Extremo izquierdo", "delantero"),
    ("576024", "Julián Álvarez", "Delantero centro", "delantero"),
    ("709187", "Nico Williams", "Extremo izquierdo", "delantero"),
]
ID = {nombre: identificador for identificador, nombre, _, _ in OFICIALES}
PLANTILLA = [{"nombre": nombre, "posicion": posicion} for _, nombre, _, posicion in OFICIALES]
NOMBRES = [jugador["nombre"] for jugador in PLANTILLA]
POSICION = {jugador["nombre"]: jugador["posicion"] for jugador in PLANTILLA}
ONCE = [n for n in NOMBRES if n not in ("Augusto Batalla", "Alejandro Balde")]

BASE = "https://www.transfermarkt.es"
URL_LIGA = f"{BASE}/laliga/startseite/wettbewerb/ES1/plus/?saison_id={TEMPORADA}"
URL_EQUIPO = f"{BASE}/fc-barcelona/kader/verein/131/saison_id/{TEMPORADA}"
PAGINA_LIGA = f"""
<table class="items"><tbody>
  <tr><td class="hauptlink"><a href="/fc-barcelona/startseite/verein/131/saison_id/{TEMPORADA}">FC Barcelona</a></td></tr>
</tbody></table>
"""  # noqa: E501


def _fila_transfermarkt(identificador, nombre, posicion, _):
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
    + "".join(_fila_transfermarkt(*oficial) for oficial in OFICIALES)
    + "</tbody></table>"
)


@dataclass
class Respuesta:
    status_code: int
    text: str


class WebFalsa:
    """Hace de cliente HTTP. Guarda las URL pedidas; una URL sin página responde 404."""

    def __init__(self, paginas):
        self.paginas = paginas
        self.urls = []

    def get(self, url, headers=None, **_):
        self.urls.append(url)
        pagina = self.paginas.get(url)
        return Respuesta(404, "") if pagina is None else Respuesta(200, pagina)


def transfermarkt():
    return WebFalsa({URL_LIGA: PAGINA_LIGA, URL_EQUIPO: PAGINA_EQUIPO})


# --- futbolfantasy.com --------------------------------------------------------------------------


def url_puntos(jornada):
    return f"https://www.futbolfantasy.com/laliga/puntos/{TEMPORADA + 1}/{jornada}/laliga-fantasy"


def pagina_puntos(jornada):
    """Una jornada ya disputada con un partido del Barcelona en el que puntúa Raphinha."""
    caja = "racha-box columna_puntos mx-auto point"
    return f"""<html><body>
<div class="col-6 col-lg-2">
    <div class="local score">1</div>
    <div class="visitante score">1</div>
</div>
<div class="fecha">
    Jornada {jornada}. Lunes, 12 de octubre del 2026 a las 21:00h
</div>
<h2 class="title">Puntos <span class="game"></span> del Barcelona</h2>
<table><tbody>
<tr class="plegado plegable">
  <td class="name" data-posicion-laliga-fantasy="Delantero">
    <i class="fa fa-plus desplegar" aria-hidden="true"></i>
    <i class="fa fa-minus plegar" aria-hidden="true"></i> Raphinha
  </td>
  <td class="data bold d-flex">
    <span class="{caja} high laliga-fantasy ">7</span><span class="{caja} low comunio scale2">4</span>
  </td>
</tr>
<tr class="desglose">
  <td colspan="3">
    <div class="desg laliga-fantasy">
      <a class="link" href="https://www.futbolfantasy.com/jugadores/raphinha/laliga-26-27">Ver la ficha del jugador</a>
    </div>
  </td>
</tr>
</tbody></table>
</body></html>"""  # noqa: E501


# Puntos por jornada. La plantilla solo admite 1-4-3-3: cambian el portero y los defensas.
# J2: recomendado 54, mejor 69 y elección simple (por la j1) 59.
# J3: recomendado 36, mejor 38 y elección simple (por la media de j1 y j2) 34.
PUNTOS = {
    1: {"Jan Oblak": 10, "Miguel Gutiérrez": 9, "Jules Koundé": 8, "Pau Cubarsí": 7,
        "Alejandro Balde": 6},
    2: {"Jan Oblak": 2, "Augusto Batalla": 9, "Jules Koundé": 4, "Pau Cubarsí": 5,
        "Dani Vivian": 6, "Miguel Gutiérrez": 3, "Alejandro Balde": 11, "Pedri": 7,
        "Federico Valverde": 1, "Lamine Yamal": 8, "Raphinha": 10, "Julián Álvarez": 2,
        "Nico Williams": 6},
    3: {"Jan Oblak": 1, "Augusto Batalla": 3, "Jules Koundé": 2, "Pau Cubarsí": 2,
        "Dani Vivian": 2, "Miguel Gutiérrez": 2, "Pedri": 5, "Federico Valverde": 5,
        "Lamine Yamal": 5, "Raphinha": 4, "Julián Álvarez": 4, "Nico Williams": 4},
}  # fmt: skip
CIFRAS_DEL_CASO = {2: (54, 69, 59), 3: (36, 38, 34)}


def escribir_puntos(datos, jornada):
    carpeta = datos / str(TEMPORADA) / f"j{jornada:02d}"
    carpeta.mkdir(parents=True, exist_ok=True)
    contenido = {
        "puntos": [
            {"jugador_id": ID[nombre], "nombre": nombre, "puntos": puntos}
            for nombre, puntos in PUNTOS[jornada].items()
        ],
        "sin_asociar": [],
    }
    (carpeta / "puntos.json").write_text(json.dumps(contenido), encoding="utf-8")


# --- Insights normalizados de una jornada ingerida ----------------------------------------------
# Pedri tiene 11 insights y Raphinha 1. Con los vectores de `OpenAIFalso`, a Pedri le llegan 10
# de los suyos y a Raphinha el suyo y 9 de Pedri: la recuperación da de media una precisión
# de (1 + 0,1) / 2 = 0,55 y una cobertura de (10/11 + 1) / 2 ≈ 0,9545.


def insight_normalizado(nombre, razon, video_id, jornada):
    return {
        "jugador": nombre,
        "jugador_id": ID[nombre],
        "equipo": "FC Barcelona",
        "posicion": POSICION[nombre],
        "nombre_transcrito": nombre,
        "accion": "alinear",
        "confianza": 0.9,
        "razon_corta": razon,
        "criterios": {
            "forma_actual": None,
            "dinamica_equipo": None,
            "calendario_rival": None,
            "prediccion_puntos": None,
        },
        "quote_original": None,
        "fuente": {
            "canal": "carrasco",
            "video_id": video_id,
            "titulo": f"Jornada {jornada}",
            "fecha": "2026-10-01",
            "jornada": jornada,
            "temporada": TEMPORADA,
            "url": f"https://youtube.com/watch?v={video_id}",
            "chunk": None,
        },
    }


def escribir_normalizados(datos, jornada):
    insights = [
        insight_normalizado("Pedri", f"Pedri suma en cada partido {n}", f"pedri{n:02d}", jornada)
        for n in range(11)
    ] + [insight_normalizado("Raphinha", "Raphinha vive en racha", "raphinha00", jornada)]
    carpeta = datos / str(TEMPORADA) / f"j{jornada:02d}"
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "insights_normalizados.json").write_text(
        json.dumps(insights, ensure_ascii=False), encoding="utf-8"
    )


# --- Conjuntos etiquetados ----------------------------------------------------------------------

# Nombres: 2 correctos, 1 resuelto a otro jugador y 2 eliminados (uno esperaba nulo).
# Precisión 2/3, cobertura 2/4 y eliminados 2/5.
CONJUNTO_NOMBRES = [
    ("Rafiña", "Rafiña está en un momento brutal", ID["Raphinha"], ID["Raphinha"]),
    ("Pedri", "Pedri tiene que jugar sí o sí", ID["Pedri"], ID["Pedri"]),
    ("Chamenamení", "Chamenamení vuelve al once", ID["Pau Cubarsí"], ID["Dani Vivian"]),
    ("Ok Bueno", "ok bueno, vamos con el siguiente", None, None),
    ("Gabi", "Gabi vuelve a entrenar", ID["Federico Valverde"], None),
]

RAZON_PEDRI = "Pedri llega con tres goles en dos partidos"
RAZON_RAPHINHA = "Raphinha marca en cada jornada"
RAZON_KOUNDE = "Koundé cierra su banda sin fallos"
RAZON_LAMINE = "Lamine Yamal vuelve tras la lesión"


def insight_extraido(jugador, razon, accion="alinear"):
    return {
        "jugador": jugador,
        "accion": accion,
        "confianza": 0.8,
        "razon_corta": razon,
        "criterios": {
            "forma_actual": None,
            "dinamica_equipo": None,
            "calendario_rival": None,
            "prediccion_puntos": None,
        },
        "quote_original": None,
    }


# Extracción: del TEXTOUNO se esperan 5 y se extraen 4, 3 de ellos correctos (precisión 0,75 y
# cobertura 0,6). El TEXTODOS no tiene ninguno.
EXTRAIDOS = {
    "TEXTOUNO": [
        insight_extraido("Pedri", RAZON_PEDRI),
        insight_extraido("Raphinha", RAZON_RAPHINHA),
        insight_extraido("Jules Koundé", RAZON_KOUNDE),
        insight_extraido("Lamine Yamal", RAZON_LAMINE),
    ],
    "TEXTODOS": [],
}
ESPERADOS = {
    "TEXTOUNO": [
        ("Pedri", "alinear"),
        ("Raphinha", "alinear"),
        ("Jules Koundé", "alinear"),
        ("Jan Oblak", "evitar"),
        ("Dani Vivian", "alinear"),
    ],
    "TEXTODOS": [],
}


def texto_del_conjunto(marca, chunk):
    return {
        "canal": "jornada_perfecta",
        "video_id": "abc123def45",
        "titulo": "ONCE IDEAL JORNADA 2",
        "fecha": "2026-09-01",
        "jornada": 2,
        "temporada": TEMPORADA,
        "url": "https://www.youtube.com/watch?v=abc123def45",
        "chunk": chunk,
        "texto": f"{marca} El analista repasa sus recomendaciones de la jornada",
        "insights_esperados": [{"jugador": j, "accion": a} for j, a in ESPERADOS[marca]],
    }


def escribir_conjunto(datos, nombre, contenido):
    fichero = datos / "evaluacion" / f"{nombre}.json"
    fichero.parent.mkdir(parents=True, exist_ok=True)
    texto = contenido if isinstance(contenido, str) else json.dumps(contenido, ensure_ascii=False)
    fichero.write_text(texto, encoding="utf-8")
    return fichero


def preparar(datos, sin=(), casos=(2,), ingeridas=None):
    """Los tres conjuntos (salvo los de `sin`), las jornadas ingeridas y sus puntos guardados."""
    if "nombres" not in sin:
        escribir_conjunto(
            datos,
            "nombres",
            {
                "temporada": TEMPORADA,
                "nombres": [
                    {"nombre_transcrito": n, "cita": c, "jugador_id": esperado}
                    for n, c, esperado, _ in CONJUNTO_NOMBRES
                ],
            },
        )
    if "extraccion" not in sin:
        escribir_conjunto(
            datos,
            "extraccion",
            {"textos": [texto_del_conjunto("TEXTOUNO", 0), texto_del_conjunto("TEXTODOS", 1)]},
        )
    if "recomendacion" not in sin:
        escribir_conjunto(
            datos,
            "recomendacion",
            {
                "casos": [
                    {"temporada": TEMPORADA, "jornada": j, "plantilla": PLANTILLA} for j in casos
                ]
            },
        )
    for jornada in casos if ingeridas is None else ingeridas:
        escribir_normalizados(datos, jornada)
    for jornada in PUNTOS:
        escribir_puntos(datos, jornada)


# --- OpenAI -------------------------------------------------------------------------------------

RESUMEN_PEDRI = "Pedri vuelve a ser titular y es el capitán"
RESUMEN_RAPHINHA = "Raphinha juega de extremo zurdo"
RESUMENES = {"Pedri": RESUMEN_PEDRI, "Raphinha": RESUMEN_RAPHINHA}

_ESCAPE_UNICODE = re.compile(r"\\+u([0-9a-fA-F]{4})")


def _respuesta(contenido):
    mensaje = SimpleNamespace(
        role="assistant",
        content=json.dumps(contenido, ensure_ascii=False),
        refusal=None,
        tool_calls=None,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=mensaje, finish_reason="stop")])


def _vector(texto):
    if "Pedri" in texto:
        return [1.0, 0.0, 0.0]
    if "Raphinha" in texto:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


class OpenAIFalso:
    """Hace de cliente de OpenAI para todas las evaluaciones.

    Cada petición de chat se atiende según el nombre del esquema de respuesta que pide. Los
    vectores separan los textos que nombran a Pedri, a Raphinha y a ninguno de los dos. Solo
    Pedri y Raphinha tienen un resumen que juzgar: los demás jugadores salen `sin_datos`.
    Cuenta las extracciones por marca de texto y las peticiones del once.
    """

    def __init__(self):
        self.extracciones = Counter()
        self.onces = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._crear))
        self.embeddings = SimpleNamespace(create=self._embeddings)

    def _embeddings(self, *, model, input, **_):
        textos = [input] if isinstance(input, str) else list(input)
        return SimpleNamespace(data=[SimpleNamespace(embedding=_vector(t)) for t in textos])

    def _crear(self, **peticion):
        sin_instrucciones = [
            mensaje
            for mensaje in peticion.get("messages", [])
            if not (isinstance(mensaje, dict) and mensaje.get("role") == "system")
        ]
        enviado = json.dumps(sin_instrucciones, ensure_ascii=False, default=str)
        enviado = _ESCAPE_UNICODE.sub(lambda escape: chr(int(escape.group(1), 16)), enviado)
        esquema = (peticion.get("response_format") or {}).get("json_schema", {}).get("name")
        if esquema == "insights_extraidos":
            marca = next(m for m in EXTRAIDOS if m in enviado)
            self.extracciones[marca] += 1
            return _respuesta({"insights": EXTRAIDOS[marca]})
        if esquema == "jugador_resuelto":
            elegido = next(obtenido for _, cita, _, obtenido in CONJUNTO_NOMBRES if cita in enviado)
            return _respuesta({"jugador_id": elegido})
        if esquema == "resumen_jugador":
            return self._resumir(enviado)
        if esquema == "recomendacion_once":
            self.onces += 1
            return _respuesta(
                {
                    "once_recomendado": [
                        {
                            "jugador": nombre,
                            "posicion": POSICION[nombre],
                            "confianza": 0.75,
                            "justificacion": "En forma",
                        }
                        for nombre in ONCE
                    ],
                    "advertencias": [],
                    "resumen": "Once de prueba",
                }
            )
        raise AssertionError(f"Petición inesperada con esquema {esquema!r}")

    def _resumir(self, enviado):
        nombrados = [n for n in NOMBRES if n in enviado]
        otros = [n for n in nombrados if n not in RESUMENES]
        if len(otros) > 1:
            raise AssertionError(f"Petición de resumen con varios jugadores: {otros}")
        sujeto = otros[0] if otros else "Raphinha" if "Raphinha" in nombrados else "Pedri"
        return _respuesta(
            {
                "jugador": sujeto,
                "posicion": POSICION[sujeto],
                "accion_recomendada": "alinear" if sujeto in RESUMENES else "sin_datos",
                "confianza_media": 0.8,
                "resumen": RESUMENES.get(sujeto, "No hay información suficiente"),
                "fuentes": ["carrasco - Jornada 2"],
            }
        )


# --- Juez de RAGAS -------------------------------------------------------------------------------

NO_RESPALDADA_INSIGHT = ("Pedri lanza los penaltis", "El texto no habla de penaltis")
NO_RESPALDADA_RESUMEN = ("Pedri es el capitán", "Ningún insight habla de la capitanía")

# Insights: Pedri 1/3, Raphinha 1, Koundé 1 y Lamine 0; media 7/12 ≈ 0,5833.
# Resúmenes: Pedri 1/2 y Raphinha 0; media 0,25.
VEREDICTOS = {
    RAZON_PEDRI: [
        ("Pedri marca goles", 1, "Lleva tres goles"),
        ("Pedri es titular indiscutible", 0, "El texto no dice que sea titular"),
        (*NO_RESPALDADA_INSIGHT[:1], 0, NO_RESPALDADA_INSIGHT[1]),
    ],
    RAZON_RAPHINHA: [("Raphinha está en racha", 1, "Marca cada jornada")],
    RAZON_KOUNDE: [("Koundé defiende bien", 1, "Cierra su banda")],
    RAZON_LAMINE: [("Lamine Yamal vuelve a jugar", 0, "El texto no nombra a Lamine Yamal")],
    RESUMEN_PEDRI: [
        ("Pedri vuelve a ser titular", 1, "Lo dicen los insights"),
        (*NO_RESPALDADA_RESUMEN[:1], 0, NO_RESPALDADA_RESUMEN[1]),
    ],
    RESUMEN_RAPHINHA: [("Raphinha juega de extremo", 0, "Ningún insight dice su demarcación")],
}


def entrada(prompt):
    """Los datos que RAGAS pone al final de su petición al juez, como diccionario."""
    return json.loads(prompt.rsplit("input: ", 1)[1].rsplit("Output:", 1)[0])


class JuezFalso(InstructorBaseRagasLLM):
    """Hace de modelo juez de RAGAS: reconoce la respuesta juzgada (una razón de insight o un
    resumen) por su texto y sus afirmaciones en la petición de veredictos."""

    def generate(self, prompt, response_model):
        raise AssertionError("RAGAS debe usar agenerate")

    async def agenerate(self, prompt, response_model):
        datos = entrada(prompt)
        if response_model is StatementGeneratorOutput:
            juzgadas = [r for r in VEREDICTOS if r in datos["answer"]]
            if len(juzgadas) != 1:
                raise AssertionError(f"Respuesta a juzgar inesperada: {datos['answer']!r}")
            return StatementGeneratorOutput(statements=[a for a, _, _ in VEREDICTOS[juzgadas[0]]])
        if response_model is NLIStatementOutput:
            veredictos = next(
                v for v in VEREDICTOS.values() if [a for a, _, _ in v] == datos["statements"]
            )
            return NLIStatementOutput(
                statements=[
                    {"statement": a, "reason": motivo, "verdict": v} for a, v, motivo in veredictos
                ]
            )
        raise AssertionError(f"Modelo de salida inesperado: {response_model}")


# --- Ejecución ----------------------------------------------------------------------------------


def servicios(cliente=None, futbolfantasy=None):
    cliente = cliente or OpenAIFalso()
    juez = JuezFalso()
    return Servicios(
        youtube=lambda clave: None,
        transcripciones=None,
        transfermarkt=transfermarkt(),
        openai=lambda clave: cliente,
        esperar=lambda segundos: None,
        futbolfantasy=futbolfantasy or WebFalsa({}),
        juez=lambda clave, modelo: juez,
        ahora=lambda: AHORA,
    )


@dataclass
class Ejecucion:
    codigo: int
    salida: str
    errores: str


def evaluar(capsys, con=None):
    """Lanza `fantasy-copilot evaluar` en el mismo proceso."""
    errores_de_salida = ""
    try:
        codigo = main(["evaluar"], servicios=con or servicios())
    except SystemExit as salida:
        codigo = salida.code
        if isinstance(codigo, str):
            errores_de_salida, codigo = codigo, 1
        codigo = codigo or 0
    capturado = capsys.readouterr()
    return Ejecucion(codigo, capturado.out, capturado.err + errores_de_salida)


def informes(datos):
    return sorted((datos / "evaluacion" / "informes").glob("*"))


def informe(datos):
    return (datos / "evaluacion" / "informes" / f"{FECHA}.md").read_text(encoding="utf-8")


_NUMERO = re.compile(r"(?<![\d.,])(\d+(?:[.,]\d+)?)([^\S\n]*%)?")


def muestra_fraccion(texto, valor):
    """Si el texto da `valor` como fracción (0.67, 0,67) o como porcentaje (67 %)."""
    numeros = [
        float(cifra.replace(",", ".")) / (100 if porcentaje else 1)
        for cifra, porcentaje in _NUMERO.findall(texto)
    ]
    return any(abs(numero - valor) <= 0.005 for numero in numeros)


def muestra_entero(texto, valor):
    return re.search(rf"(?<![\d.,]){valor}(?!\d|[.,]\d)", texto) is not None


def muestra(texto, valor):
    return (
        muestra_entero(texto, valor) if isinstance(valor, int) else muestra_fraccion(texto, valor)
    )


def seccion(texto, palabra):
    """El texto bajo el primer encabezado Markdown que contiene `palabra`, hasta el siguiente
    encabezado de su nivel o de uno superior."""
    lineas = texto.splitlines()
    nivel = None
    for numero, linea in enumerate(lineas):
        marca = re.match(r"\s*(#+)\s", linea)
        if marca and palabra.lower() in linea.lower():
            nivel, inicio = len(marca.group(1)), numero + 1
            break
    assert nivel is not None, f"El informe no tiene ningún encabezado con {palabra!r}:\n{texto}"
    bloque = []
    for linea in lineas[inicio:]:
        marca = re.match(r"\s*(#+)\s", linea)
        if marca and len(marca.group(1)) <= nivel:
            break
        bloque.append(linea)
    return "\n".join(bloque)


def cifras_de(texto, etiqueta):
    """Las cifras que acompañan a `etiqueta`: `5 nombres`, `Nombres: 5` o `| Nombres | 5 |`."""
    espacio = r"[^\S\n]"
    patron = (
        rf"(\d+){espacio}+(?:\w+{espacio}+)?{etiqueta}"
        rf"|{etiqueta}\w*(?:{espacio}+\w+)?{espacio}*[:|]{espacio}*(\d+)"
    )
    return [int(a or b) for a, b in re.findall(patron, texto, re.IGNORECASE)]


CIFRAS = [
    pytest.param(2 / 3, id="precision_de_los_nombres"),
    pytest.param(0.5, id="cobertura_de_los_nombres"),
    pytest.param(0.4, id="eliminados_de_los_nombres"),
    pytest.param(0.75, id="precision_de_la_extraccion"),
    pytest.param(0.6, id="cobertura_de_la_extraccion"),
    pytest.param(7 / 12, id="fidelidad_media_de_los_insights"),
    pytest.param(0.55, id="precision_media_de_la_recuperacion"),
    pytest.param(21 / 22, id="cobertura_media_de_la_recuperacion"),
    pytest.param(54, id="puntos_del_once_recomendado"),
    pytest.param(69, id="puntos_del_mejor_once"),
    pytest.param(59, id="puntos_de_la_eleccion_simple"),
    pytest.param(0.25, id="fidelidad_media_de_los_resumenes"),
]


# --- Tests --------------------------------------------------------------------------------------


def test_sin_ningun_conjunto_termina_con_codigo_1_nombrando_los_ficheros_y_sin_informe(
    capsys, datos
):
    resultado = evaluar(capsys)

    assert resultado.codigo == 1
    faltan = [
        fichero
        for fichero in ("nombres.json", "extraccion.json", "recomendacion.json")
        if fichero not in resultado.salida + resultado.errores
    ]
    assert faltan == []
    assert informes(datos) == []


def test_un_conjunto_danado_termina_con_codigo_1_con_su_mensaje_y_sin_informe(capsys, datos):
    preparar(datos)
    escribir_conjunto(datos, "extraccion", {"textos": "no es una lista"})
    with pytest.raises(ErrorConjunto) as error:
        evaluar_extraccion(datos_dir=datos, modelo="modelo", cliente=OpenAIFalso())

    resultado = evaluar(capsys)

    assert resultado.codigo == 1
    assert str(error.value) in resultado.errores
    assert informes(datos) == []


def test_con_los_tres_conjuntos_termina_con_codigo_0_y_escribe_el_informe_con_la_fecha_del_dia(
    capsys, datos
):
    preparar(datos)

    resultado = evaluar(capsys)

    assert resultado.codigo == 0
    assert [ruta.name for ruta in informes(datos)] == [f"{FECHA}.md"]


def test_el_informe_de_un_dia_sustituye_al_que_ya_habia_de_ese_dia(capsys, datos):
    preparar(datos)
    carpeta = datos / "evaluacion" / "informes"
    carpeta.mkdir(parents=True)
    (carpeta / f"{FECHA}.md").write_text("INFORME ANTERIOR DEL MISMO DIA", encoding="utf-8")

    evaluar(capsys)

    assert [ruta.name for ruta in informes(datos)] == [f"{FECHA}.md"]
    assert "INFORME ANTERIOR" not in informe(datos)


@pytest.mark.parametrize("valor", CIFRAS)
def test_la_salida_muestra_la_cifra_de_cada_evaluacion(capsys, datos, valor):
    preparar(datos)

    resultado = evaluar(capsys)

    assert muestra(resultado.salida, valor), resultado.salida


@pytest.mark.parametrize("valor", CIFRAS)
def test_el_informe_recoge_la_cifra_de_cada_evaluacion(capsys, datos, valor):
    preparar(datos)

    evaluar(capsys)

    assert muestra(informe(datos), valor), informe(datos)


@pytest.mark.parametrize(
    "etiqueta, elementos", [("nombre", 5), ("texto", 2), ("caso", 2)], ids=lambda x: str(x)
)
def test_el_informe_da_el_numero_de_elementos_de_cada_conjunto(capsys, datos, etiqueta, elementos):
    preparar(datos, casos=(2, 3))

    evaluar(capsys)

    assert elementos in cifras_de(informe(datos), etiqueta), informe(datos)


def test_el_informe_da_los_modelos_de_la_configuracion(capsys, datos):
    preparar(datos)

    evaluar(capsys)

    assert [m for m in MODELOS.values() if m not in informe(datos)] == []


@pytest.mark.parametrize("jornada", [2, 3])
def test_el_informe_da_una_fila_por_caso_con_jornada_temporada_puntos_y_fidelidad(
    capsys, datos, jornada
):
    preparar(datos, casos=(2, 3))

    evaluar(capsys)

    cifras = (jornada, TEMPORADA, *CIFRAS_DEL_CASO[jornada])
    filas = [
        linea
        for linea in informe(datos).splitlines()
        if all(muestra_entero(linea, cifra) for cifra in cifras) and muestra_fraccion(linea, 0.25)
    ]
    assert len(filas) == 1, informe(datos)


@pytest.mark.parametrize(
    "afirmacion, motivo",
    [NO_RESPALDADA_INSIGHT, NO_RESPALDADA_RESUMEN],
    ids=["de_un_insight", "de_un_resumen"],
)
def test_el_informe_da_aparte_cada_afirmacion_no_respaldada_con_su_motivo(
    capsys, datos, afirmacion, motivo
):
    preparar(datos)

    evaluar(capsys)

    no_respaldadas = seccion(informe(datos), "respaldad")
    assert afirmacion in no_respaldadas
    assert motivo in no_respaldadas


# Por conjunto: el encabezado de su sección y una cifra que solo sale si se evalúa.
CONJUNTOS = {
    "nombres": ("nombres", 2 / 3),
    "extraccion": ("extracción", 0.75),
    "recomendacion": ("recomendación", 54),
}


@pytest.mark.parametrize("ausente", list(CONJUNTOS))
def test_si_falta_un_conjunto_termina_con_codigo_0_y_escribe_el_informe(capsys, datos, ausente):
    preparar(datos, sin=(ausente,))

    resultado = evaluar(capsys)

    assert resultado.codigo == 0, resultado.errores
    assert [ruta.name for ruta in informes(datos)] == [f"{FECHA}.md"]


@pytest.mark.parametrize("ausente", list(CONJUNTOS))
def test_si_falta_un_conjunto_los_demas_se_evaluan_e_informan(capsys, datos, ausente):
    preparar(datos, sin=(ausente,))

    evaluar(capsys)

    sin_cifra = [
        conjunto
        for conjunto, (_, cifra) in CONJUNTOS.items()
        if conjunto != ausente and not muestra(informe(datos), cifra)
    ]
    assert sin_cifra == [], informe(datos)


@pytest.mark.parametrize("ausente", list(CONJUNTOS))
def test_si_falta_un_conjunto_su_seccion_del_informe_dice_que_no_hay_conjunto(
    capsys, datos, ausente
):
    preparar(datos, sin=(ausente,))

    evaluar(capsys)

    encabezado, _ = CONJUNTOS[ausente]
    assert "no hay conjunto" in seccion(informe(datos), encabezado).lower()


def test_un_caso_de_una_jornada_sin_ingerir_avisa_de_la_ingesta_que_hay_que_ejecutar(capsys, datos):
    preparar(datos, casos=(2, 5), ingeridas=(2,))

    resultado = evaluar(capsys)

    assert "ingesta --jornada 5 --temporada 2026" in resultado.salida + resultado.errores


def test_un_caso_de_una_jornada_sin_ingerir_no_impide_evaluar_los_demas(capsys, datos):
    preparar(datos, casos=(5, 2), ingeridas=(2,))

    resultado = evaluar(capsys)

    assert resultado.codigo == 0
    filas = [
        linea
        for linea in informe(datos).splitlines()
        if all(muestra_entero(linea, cifra) for cifra in CIFRAS_DEL_CASO[2])
    ]
    assert filas, informe(datos)


def test_cada_texto_de_extraccion_se_extrae_una_sola_vez(capsys, datos):
    preparar(datos)
    cliente = OpenAIFalso()

    evaluar(capsys, con=servicios(cliente=cliente))

    assert cliente.extracciones == {"TEXTOUNO": 1, "TEXTODOS": 1}


def test_cada_caso_de_recomendacion_se_recomienda_una_sola_vez(capsys, datos):
    preparar(datos, casos=(2, 3))
    cliente = OpenAIFalso()

    evaluar(capsys, con=servicios(cliente=cliente))

    assert cliente.onces == 2


def test_se_descargan_los_puntos_que_faltan_de_la_jornada_y_las_anteriores_y_no_los_guardados(
    capsys, datos
):
    preparar(datos, casos=(3,))
    for jornada in (2, 3):
        (datos / str(TEMPORADA) / f"j{jornada:02d}" / "puntos.json").unlink()
    web = WebFalsa({url_puntos(2): pagina_puntos(2), url_puntos(3): pagina_puntos(3)})

    evaluar(capsys, con=servicios(futbolfantasy=web))

    assert sorted(web.urls) == [url_puntos(2), url_puntos(3)]


# --- Conjuntos de nombres y de recomendación dañados --------------------------------------------
# Cada uno con lo que debe nombrar el mensaje de error.

DANADOS = [
    pytest.param(
        "nombres",
        {"temporada": TEMPORADA, "nombres": "no es una lista"},
        [r"nombres\.json"],
        id="nombres_sin_lista",
    ),
    pytest.param(
        "recomendacion",
        {"casos": [{"temporada": TEMPORADA, "plantilla": []}]},
        [r"recomendacion\.json", r"caso\W{0,3}1\b"],
        id="caso_sin_jornada",
    ),
    pytest.param(
        "recomendacion",
        {
            "casos": [
                {
                    "temporada": TEMPORADA,
                    "jornada": 2,
                    "plantilla": [
                        *PLANTILLA,
                        {"nombre": "Jugador Inventado", "posicion": "defensa"},
                    ],
                }
            ]
        },
        [r"caso\W{0,3}1\b", r"Jugador Inventado"],
        id="jugador_fuera_de_la_lista_oficial",
    ),
]


class OpenAIQueCuenta(OpenAIFalso):
    """Cuenta todas las peticiones a OpenAI: de chat, de embeddings y del juez de RAGAS."""

    def __init__(self):
        super().__init__()
        self.peticiones = 0
        crear, vectorizar = self._crear, self._embeddings
        self.chat.completions.create = lambda **p: self._contar(crear, **p)
        self.embeddings.create = lambda **p: self._contar(vectorizar, **p)

    def _contar(self, peticion, **argumentos):
        self.peticiones += 1
        return peticion(**argumentos)


class JuezQueCuenta(JuezFalso):
    def __init__(self, cliente):
        super().__init__()
        self.cliente = cliente

    async def agenerate(self, prompt, response_model):
        self.cliente.peticiones += 1
        return await super().agenerate(prompt, response_model)


@pytest.mark.parametrize("conjunto, contenido, nombrados", DANADOS)
def test_un_conjunto_de_nombres_o_recomendacion_danado_termina_con_codigo_1_nombrado_y_sin_informe(
    capsys, datos, conjunto, contenido, nombrados
):
    preparar(datos)
    escribir_conjunto(datos, conjunto, contenido)

    resultado = evaluar(capsys)

    assert resultado.codigo == 1
    mensaje = resultado.salida + resultado.errores
    assert [n for n in nombrados if not re.search(n, mensaje, re.IGNORECASE)] == [], mensaje
    assert informes(datos) == []


@pytest.mark.parametrize("conjunto, contenido, nombrados", DANADOS)
def test_un_conjunto_de_nombres_o_recomendacion_danado_no_hace_ninguna_peticion_a_openai(
    capsys, datos, conjunto, contenido, nombrados
):
    preparar(datos)
    escribir_conjunto(datos, conjunto, contenido)
    cliente = OpenAIQueCuenta()
    con = servicios(cliente=cliente)
    juez = JuezQueCuenta(cliente)
    con.juez = lambda clave, modelo: juez

    evaluar(capsys, con=con)

    assert cliente.peticiones == 0


# --- Un caso con puntos provisionales -----------------------------------------------------------
# La jornada 3 acabó el 20 de octubre a las 21:00 y se evalúa el 21 a las 12:00 (`AHORA`): sus
# puntos aún son provisionales.


def preparar_jornada_3_provisional(datos):
    preparar(datos, casos=(2, 3))
    (datos / str(TEMPORADA) / "j03" / "puntos.json").unlink()
    provisional = pagina_puntos(3).replace(
        "Lunes, 12 de octubre del 2026", "Martes, 20 de octubre del 2026"
    )
    return WebFalsa({url_puntos(3): provisional})


def test_un_caso_con_puntos_provisionales_no_impide_evaluar_los_demas(capsys, datos):
    web = preparar_jornada_3_provisional(datos)

    resultado = evaluar(capsys, con=servicios(futbolfantasy=web))

    assert resultado.codigo == 0, resultado.errores
    filas = [
        linea
        for linea in informe(datos).splitlines()
        if all(muestra_entero(linea, cifra) for cifra in CIFRAS_DEL_CASO[2])
    ]
    assert filas, informe(datos)


def test_un_caso_con_puntos_provisionales_avisa_de_que_se_salta_por_ser_provisionales(
    capsys, datos
):
    web = preparar_jornada_3_provisional(datos)

    resultado = evaluar(capsys, con=servicios(futbolfantasy=web))

    avisos = [
        linea
        for linea in (resultado.salida + resultado.errores).splitlines()
        if "provisional" in linea.lower() and muestra_entero(linea, 3)
    ]
    assert avisos, resultado.salida + resultado.errores


def test_un_caso_con_puntos_provisionales_sale_en_el_informe_como_no_evaluado(capsys, datos):
    web = preparar_jornada_3_provisional(datos)

    evaluar(capsys, con=servicios(futbolfantasy=web))

    no_evaluado = re.compile(r"provisional|sin evaluar|no evaluad|saltad", re.IGNORECASE)
    encabezado, saltados = "", []
    for linea in informe(datos).splitlines():
        if re.match(r"\s*#+\s", linea):
            encabezado = linea
        elif muestra_entero(linea, 3) and muestra_entero(linea, TEMPORADA):
            if no_evaluado.search(linea) or no_evaluado.search(encabezado):
                saltados.append(linea)
    assert saltados, informe(datos)


def test_un_caso_con_puntos_provisionales_no_pide_la_recomendacion_del_once(capsys, datos):
    web = preparar_jornada_3_provisional(datos)
    cliente = OpenAIFalso()

    evaluar(capsys, con=servicios(cliente=cliente, futbolfantasy=web))

    assert cliente.onces == 1


def test_los_puntos_provisionales_de_la_jornada_no_se_guardan(capsys, datos):
    web = preparar_jornada_3_provisional(datos)

    evaluar(capsys, con=servicios(futbolfantasy=web))

    assert not (datos / str(TEMPORADA) / "j03" / "puntos.json").exists()
