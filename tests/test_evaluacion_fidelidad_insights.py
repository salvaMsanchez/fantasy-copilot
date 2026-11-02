import json
from types import SimpleNamespace

import openai
import pytest
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections.faithfulness.util import (
    NLIStatementOutput,
    StatementGeneratorOutput,
)

from fantasy_copilot.evaluacion import evaluar_extraccion, evaluar_fidelidad_insights

VIDEO_ID = "abc123def45"

TEXTO_UNO = "TROZO_UNO El analista dice que Pedri lleva tres goles en dos partidos"
TEXTO_DOS = "TROZO_DOS El analista dice que Raphinha ha marcado en las tres ultimas jornadas"

RAZON_PEDRI = "Pedri atraviesa un gran momento goleador"
RAZON_RAPHINHA = "Raphinha no para de marcar"

# Pedri: 4 afirmaciones, 3 respaldadas. Raphinha: 2 afirmaciones, las 2 respaldadas.
VEREDICTOS = {
    RAZON_PEDRI: [
        ("Pedri atraviesa un gran momento", 1, "Lleva tres goles en dos partidos"),
        ("Pedri marca goles", 1, "Lleva tres goles"),
        ("Pedri juega de centrocampista", 1, "Se deduce del texto"),
        ("Pedri es el lanzador de penaltis", 0, "El texto no habla de penaltis"),
    ],
    RAZON_RAPHINHA: [
        ("Raphinha marca goles", 1, "Ha marcado en tres jornadas"),
        ("Raphinha esta en racha", 1, "Ha marcado en las tres ultimas jornadas"),
    ],
}


def texto(contenido, chunk):
    """Un texto del conjunto de extracción. Su contenido empieza por una marca única."""
    return {
        "canal": "jornada_perfecta",
        "video_id": VIDEO_ID,
        "titulo": "ONCE IDEAL JORNADA 8",
        "fecha": "2026-10-16",
        "jornada": 8,
        "temporada": 2026,
        "url": f"https://www.youtube.com/watch?v={VIDEO_ID}",
        "chunk": chunk,
        "texto": contenido,
        "insights_esperados": [],
    }


def insight(jugador, razon, **criterios):
    return {
        "jugador": jugador,
        "accion": "alinear",
        "confianza": 0.8,
        "razon_corta": razon,
        "criterios": {
            "forma_actual": None,
            "dinamica_equipo": None,
            "calendario_rival": None,
            "prediccion_puntos": None,
            **criterios,
        },
        "quote_original": None,
    }


def completa(*insights):
    contenido = json.dumps({"insights": list(insights)}, ensure_ascii=False)
    mensaje = SimpleNamespace(role="assistant", content=contenido, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=mensaje, finish_reason="stop")])


class OpenAIFalso:
    """Hace de cliente de OpenAI de la extracción: `reglas` asocia una marca del texto con la
    respuesta."""

    def __init__(self, reglas):
        self.reglas = reglas
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._crear))

    def _crear(self, **peticion):
        enviado = json.dumps(peticion.get("messages"), ensure_ascii=False, default=str)
        respuesta = next((r for marca, r in self.reglas if marca in enviado), None)
        if respuesta is None:
            raise AssertionError("Petición inesperada: ninguna regla coincide con su texto")
        return respuesta


def entrada(prompt):
    """Los datos que RAGAS pone al final de su petición al juez, como diccionario."""
    return json.loads(prompt.rsplit("input: ", 1)[1].rsplit("Output:", 1)[0])


class JuezFalso(InstructorBaseRagasLLM):
    """Hace de modelo juez de RAGAS con veredictos fijados por `razon_corta`.

    Reconoce el insight por su `razon_corta` en la respuesta que se le pide descomponer y por
    sus afirmaciones en la petición de veredictos. `fallos` asocia una `razon_corta` con la
    excepción que se lanza al juzgarla. Guarda los datos de cada petición.
    """

    def __init__(self, veredictos=None, fallos=None):
        self.veredictos = VEREDICTOS if veredictos is None else veredictos
        self.fallos = fallos or {}
        self.peticiones = []

    def generate(self, prompt, response_model):
        raise AssertionError("RAGAS debe usar agenerate")

    async def agenerate(self, prompt, response_model):
        datos = entrada(prompt)
        self.peticiones.append((response_model, datos))
        if response_model is StatementGeneratorOutput:
            razones = [r for r in {**self.veredictos, **self.fallos} if r in datos["answer"]]
            if len(razones) != 1:
                raise AssertionError(f"Respuesta a juzgar inesperada: {datos['answer']!r}")
            if razones[0] in self.fallos:
                raise self.fallos[razones[0]]
            return StatementGeneratorOutput(
                statements=[a for a, _, _ in self.veredictos[razones[0]]]
            )
        if response_model is NLIStatementOutput:
            afirmaciones = next(
                v for v in self.veredictos.values()
                if [a for a, _, _ in v] == datos["statements"]
            )  # fmt: skip
            return NLIStatementOutput(
                statements=[
                    {"statement": a, "reason": motivo, "verdict": v}
                    for a, v, motivo in afirmaciones
                ]
            )
        raise AssertionError(f"Modelo de salida inesperado: {response_model}")

    def respuestas_juzgadas(self):
        return [d["answer"] for m, d in self.peticiones if m is StatementGeneratorOutput]

    def contexto_de(self, afirmacion):
        (contexto,) = [
            d["context"]
            for m, d in self.peticiones
            if m is NLIStatementOutput and afirmacion in d["statements"]
        ]
        return contexto


def escribir_conjunto(datos_dir, textos):
    fichero = datos_dir / "evaluacion" / "extraccion.json"
    fichero.parent.mkdir(parents=True, exist_ok=True)
    fichero.write_text(json.dumps({"textos": textos}, ensure_ascii=False), encoding="utf-8")


def extraer(datos_dir, reglas):
    """Hace la extracción de un conjunto de dos trozos del mismo vídeo con un cliente falso."""
    escribir_conjunto(datos_dir, [texto(TEXTO_UNO, 0), texto(TEXTO_DOS, 1)])
    return evaluar_extraccion(
        datos_dir=datos_dir,
        modelo="modelo-de-prueba",
        cliente=OpenAIFalso(reglas),
        esperar=lambda segundos: None,
    )


REGLAS = [
    ("TROZO_UNO", completa(insight("Pedri", RAZON_PEDRI))),
    ("TROZO_DOS", completa(insight("Raphinha", RAZON_RAPHINHA))),
]


def evaluar(datos_dir, juez=None, reglas=REGLAS):
    extraccion = extraer(datos_dir, reglas)
    return evaluar_fidelidad_insights(extraccion, juez=juez or JuezFalso())


def detalle_de(resultado, jugador):
    return next(i for i in resultado["insights"] if i["jugador"] == jugador)


def test_la_fidelidad_de_un_insight_es_la_proporcion_de_afirmaciones_respaldadas(tmp_path):
    resultado = evaluar(tmp_path)

    fidelidades = {j: detalle_de(resultado, j)["fidelidad"] for j in ("Pedri", "Raphinha")}
    assert fidelidades == {"Pedri": pytest.approx(0.75), "Raphinha": pytest.approx(1.0)}


def test_la_media_es_la_de_las_fidelidades_de_los_insights(tmp_path):
    assert evaluar(tmp_path)["media"] == pytest.approx(0.875)


def test_el_detalle_identifica_el_video_y_el_trozo_de_cada_insight(tmp_path):
    resultado = evaluar(tmp_path)

    origen = {
        j: (detalle_de(resultado, j)["video_id"], detalle_de(resultado, j)["chunk"])
        for j in ("Pedri", "Raphinha")
    }
    assert origen == {"Pedri": (VIDEO_ID, 0), "Raphinha": (VIDEO_ID, 1)}


def test_el_detalle_da_cada_afirmacion_no_respaldada_con_el_motivo_del_juez(tmp_path):
    assert detalle_de(evaluar(tmp_path), "Pedri")["no_respaldadas"] == [
        {
            "afirmacion": "Pedri es el lanzador de penaltis",
            "motivo": "El texto no habla de penaltis",
        }
    ]


def test_con_los_criterios_nulos_el_juez_recibe_solo_la_razon_corta(tmp_path):
    juez = JuezFalso(veredictos={"Está en racha": [("Pedri esta en racha", 1, "Tres goles")]})
    reglas = [
        ("TROZO_UNO", completa(insight("Pedri", "Está en racha"))),
        ("TROZO_DOS", completa()),
    ]

    evaluar(tmp_path, juez=juez, reglas=reglas)

    assert juez.respuestas_juzgadas() == ["Está en racha"]


def test_el_juez_recibe_tambien_los_criterios_no_nulos(tmp_path):
    forma = "Lleva tres goles en dos partidos"
    calendario = "Juega en casa ante el colista"
    juez = JuezFalso(veredictos={RAZON_PEDRI: VEREDICTOS[RAZON_PEDRI]})
    pedri = insight("Pedri", RAZON_PEDRI, forma_actual=forma, calendario_rival=calendario)
    reglas = [("TROZO_UNO", completa(pedri)), ("TROZO_DOS", completa())]

    evaluar(tmp_path, juez=juez, reglas=reglas)

    (respuesta,) = juez.respuestas_juzgadas()
    assert [t for t in (RAZON_PEDRI, forma, calendario) if t in respuesta] == [
        RAZON_PEDRI,
        forma,
        calendario,
    ]


def test_el_contexto_del_juez_es_solo_el_texto_del_trozo_del_que_salio_el_insight(tmp_path):
    juez = JuezFalso()

    evaluar(tmp_path, juez=juez)

    assert juez.contexto_de("Raphinha marca goles") == TEXTO_DOS


def test_el_juez_recibe_dos_peticiones_por_insight_juzgado_y_ninguna_mas(tmp_path):
    juez = JuezFalso()

    evaluar(tmp_path, juez=juez)

    assert len(juez.peticiones) == 4


def test_un_insight_del_que_el_juez_no_saca_afirmaciones_queda_sin_valor_y_fuera_de_la_media(
    tmp_path,
):
    juez = JuezFalso(veredictos={**VEREDICTOS, RAZON_RAPHINHA: []})

    resultado = evaluar(tmp_path, juez=juez)

    assert (detalle_de(resultado, "Raphinha")["fidelidad"], resultado["media"]) == (
        None,
        pytest.approx(0.75),
    )


def test_si_ningun_insight_tiene_valor_la_media_es_nula(tmp_path):
    juez = JuezFalso(veredictos={RAZON_PEDRI: [], RAZON_RAPHINHA: []})

    assert evaluar(tmp_path, juez=juez)["media"] is None


def test_si_no_se_extrajo_ningun_insight_la_media_es_nula(tmp_path):
    reglas = [("TROZO_UNO", completa()), ("TROZO_DOS", completa())]

    assert evaluar(tmp_path, reglas=reglas)["media"] is None


def test_si_el_juez_falla_para_un_insight_queda_con_el_error(tmp_path):
    juez = JuezFalso(fallos={RAZON_PEDRI: openai.OpenAIError("fallo de conexión")})

    pedri = detalle_de(evaluar(tmp_path, juez=juez), "Pedri")

    assert (pedri["estado"], "fallo de conexión" in pedri["error"]) == ("error", True)


def test_si_el_juez_falla_para_un_insight_los_demas_se_juzgan_y_solo_ellos_hacen_la_media(
    tmp_path,
):
    juez = JuezFalso(fallos={RAZON_PEDRI: openai.OpenAIError("fallo de conexión")})

    assert evaluar(tmp_path, juez=juez)["media"] == pytest.approx(1.0)
