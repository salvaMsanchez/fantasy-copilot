import json
from types import SimpleNamespace

import openai
import pytest
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections.faithfulness.util import (
    NLIStatementOutput,
    StatementGeneratorOutput,
)

from fantasy_copilot.agente import recomendar_once
from fantasy_copilot.evaluacion import evaluar_fidelidad

JORNADA = 8
TEMPORADA = 2026

PLANTILLA = [
    {"nombre": nombre, "jugador_id": jugador_id, "equipo": equipo, "posicion": posicion}
    for nombre, jugador_id, equipo, posicion in [
        ("Pedri", "683840", "FC Barcelona", "centrocampista"),
        ("Raphinha", "411295", "FC Barcelona", "delantero"),
        ("Jan Oblak", "121483", "Atletico de Madrid", "portero"),
    ]
]
NOMBRES = [j["nombre"] for j in PLANTILLA]
POSICION = {j["nombre"]: j["posicion"] for j in PLANTILLA}

RESUMEN_PEDRI = "Pedri vuelve a ser titular y llega en buena forma ante un rival flojo"
RESUMEN_RAPHINHA = "Raphinha encadena tres partidos marcando"

# Pedri: 4 afirmaciones, 3 respaldadas. Raphinha: 2 afirmaciones, las 2 respaldadas.
VEREDICTOS = {
    RESUMEN_PEDRI: [
        ("Pedri vuelve a ser titular", 1, "Lo dice el primer insight"),
        ("Pedri llega en buena forma", 1, "Lo dice el primer insight"),
        ("Pedri juega ante un rival flojo", 1, "Lo dice el segundo insight"),
        ("Pedri es el lanzador de penaltis", 0, "Ningun insight habla de penaltis"),
    ],
    RESUMEN_RAPHINHA: [
        ("Raphinha encadena tres partidos", 1, "Lo dice el insight"),
        ("Raphinha esta marcando", 1, "Lo dice el insight"),
    ],
}

TEXTO_PEDRI = "Pedri recupera la titularidad tras la lesion segun carrasco"
TEXTO_PEDRI_RIVAL = "Pedri se mide a un rival que encaja mucho segun jornadaperfecta"
TEXTO_RAPHINHA = "Raphinha lleva tres jornadas seguidas viendo puerta segun carrasco"


def insight(jugador, texto, confianza=0.9, jornada=JORNADA, canal="carrasco"):
    """Un resultado de `Indice.consultar` sobre `jugador`."""
    return {
        "id": f"{jugador}-{texto}",
        "texto": texto,
        "metadatos": {
            "jugador": jugador,
            "jugador_id": "0",
            "equipo": "FC Barcelona",
            "posicion": POSICION[jugador],
            "accion": "alinear",
            "confianza": confianza,
            "canal": canal,
            "fecha": "2026-10-16",
            "jornada": jornada,
            "temporada": TEMPORADA,
            "url": f"https://youtube.com/watch?v={canal}",
        },
        "distancia": 0.1,
    }


INSIGHTS = [
    insight("Pedri", TEXTO_PEDRI),
    insight("Pedri", TEXTO_PEDRI_RIVAL, canal="jornadaperfecta"),
    insight("Raphinha", TEXTO_RAPHINHA),
]


class IndiceFalso:
    """Hace de índice: devuelve los insights de la jornada cuyo jugador aparece en la pregunta."""

    def __init__(self, insights):
        self.insights = list(insights)

    def consultar(self, pregunta, *, jornada, temporada, top_k=10):
        return [
            i
            for i in self.insights
            if i["metadatos"]["jugador"] in pregunta
            and i["metadatos"]["jornada"] == jornada
            and i["metadatos"]["temporada"] == temporada
        ][:top_k]


def _respuesta(contenido):
    mensaje = SimpleNamespace(
        role="assistant",
        content=json.dumps(contenido, ensure_ascii=False),
        refusal=None,
        tool_calls=None,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=mensaje, finish_reason="stop")])


RESUMEN_OBLAK = "Los insights hablan de otros porteros y no se puede decidir sobre Oblak"
RESUMENES = {"Pedri": RESUMEN_PEDRI, "Raphinha": RESUMEN_RAPHINHA, "Jan Oblak": RESUMEN_OBLAK}

# Insights de Oblak que no permiten decidir: el modelo responde sin_datos aunque los reciba.
INSIGHTS_OBLAK = [
    insight("Jan Oblak", "Courtois para un penalti en el ultimo minuto segun carrasco"),
    insight("Jan Oblak", "Ter Stegen vuelve a entrenar con el grupo segun jornadaperfecta"),
]

# Si el resumen de Oblak se juzgara, el juez le daría 1 de 2 afirmaciones respaldadas.
VEREDICTOS_CON_OBLAK = {
    **VEREDICTOS,
    RESUMEN_OBLAK: [
        ("Los insights hablan de otros porteros", 1, "Hablan de Courtois y Ter Stegen"),
        ("No se puede decidir sobre Oblak", 0, "Los insights recomiendan alinear"),
    ],
}


class OpenAIRecomendacion:
    """Hace de cliente de OpenAI de la recomendación: una petición que nombra a un solo jugador
    es su resumen; si nombra a más, es la del once. `acciones` fija la `accion_recomendada` de
    cada jugador; por defecto, `alinear`."""

    def __init__(self, acciones=None):
        self.acciones = acciones or {}
        self.peticiones = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._crear))

    def _crear(self, **peticion):
        self.peticiones += 1
        enviado = str(peticion.get("messages"))
        nombrados = [n for n in NOMBRES if n in enviado]
        if len(nombrados) == 1:
            nombre = nombrados[0]
            return _respuesta(
                {
                    "jugador": nombre,
                    "posicion": POSICION[nombre],
                    "accion_recomendada": self.acciones.get(nombre, "alinear"),
                    "confianza_media": 0.8,
                    "resumen": RESUMENES[nombre],
                    "fuentes": ["carrasco - Jornada 8"],
                }
            )
        return _respuesta(
            {
                "once_recomendado": [
                    {
                        "jugador": n,
                        "posicion": POSICION[n],
                        "confianza": 0.8,
                        "justificacion": "En racha",
                    }
                    for n in NOMBRES
                ],
                "advertencias": [],
                "resumen": "Once de prueba.",
            }
        )


class JuezFalso(InstructorBaseRagasLLM):
    """Hace de modelo juez de RAGAS con veredictos fijados por resumen.

    Reconoce el resumen por su texto en la petición de afirmaciones y por sus afirmaciones en la
    de veredictos. `veredictos` asocia cada resumen con sus afirmaciones; `fallos`, con la
    excepción que se lanza al juzgarlo. Un resumen desconocido es un error del test.
    """

    def __init__(self, veredictos=None, fallos=None):
        self.veredictos = VEREDICTOS if veredictos is None else veredictos
        self.fallos = fallos or {}
        self.peticiones = []

    def generate(self, prompt, response_model):
        raise AssertionError("RAGAS debe usar agenerate")

    async def agenerate(self, prompt, response_model):
        self.peticiones.append((response_model, prompt))
        if response_model is StatementGeneratorOutput:
            resumen = self._resumen_nombrado(prompt)
            if resumen in self.fallos:
                raise self.fallos[resumen]
            return StatementGeneratorOutput(
                statements=[afirmacion for afirmacion, _, _ in self.veredictos[resumen]]
            )
        if response_model is NLIStatementOutput:
            resumen = next(
                r for r, afirmaciones in self.veredictos.items()
                if afirmaciones and afirmaciones[0][0] in prompt
            )  # fmt: skip
            return NLIStatementOutput(
                statements=[
                    {"statement": a, "reason": motivo, "verdict": v}
                    for a, v, motivo in self.veredictos[resumen]
                ]
            )
        raise AssertionError(f"Modelo de salida inesperado: {response_model}")

    def _resumen_nombrado(self, prompt):
        nombrados = [r for r in {**self.veredictos, **self.fallos} if r in prompt]
        if len(nombrados) != 1:
            raise AssertionError(f"Petición al juez con resúmenes inesperados: {nombrados}")
        return nombrados[0]

    def peticiones_de_veredicto(self, afirmacion):
        return [
            p for modelo, p in self.peticiones if modelo is NLIStatementOutput and afirmacion in p
        ]


def recomendar(insights=INSIGHTS, acciones=None):
    return recomendar_once(
        PLANTILLA,
        jornada=JORNADA,
        temporada=TEMPORADA,
        indice=IndiceFalso(insights),
        modelo="modelo-de-prueba",
        cliente=OpenAIRecomendacion(acciones),
        esperar=lambda segundos: None,
    )


def evaluar(juez=None):
    return evaluar_fidelidad(recomendar(), juez=juez or JuezFalso())


def detalle_de(resultado, nombre):
    return next(j for j in resultado["jugadores"] if j["jugador"] == nombre)


def test_la_fidelidad_de_un_resumen_es_la_proporcion_de_afirmaciones_respaldadas():
    resultado = evaluar()

    fidelidades = {n: detalle_de(resultado, n)["fidelidad"] for n in ("Pedri", "Raphinha")}
    assert fidelidades == {"Pedri": pytest.approx(0.75), "Raphinha": pytest.approx(1.0)}


def test_la_media_es_la_de_las_fidelidades_de_los_resumenes_juzgados():
    # Oblak, sin insights, tiene resumen sin_datos: si contara, la media no sería 0,875.
    assert evaluar()["media"] == pytest.approx(0.875)


def test_el_detalle_da_el_resumen_que_se_juzgo():
    assert detalle_de(evaluar(), "Pedri")["resumen"] == RESUMEN_PEDRI


def test_el_detalle_da_cada_afirmacion_no_respaldada_con_el_motivo_del_juez():
    assert detalle_de(evaluar(), "Pedri")["no_respaldadas"] == [
        {
            "afirmacion": "Pedri es el lanzador de penaltis",
            "motivo": "Ningun insight habla de penaltis",
        }
    ]


def test_un_jugador_sin_datos_no_se_juzga_y_aparece_como_sin_datos():
    oblak = detalle_de(evaluar(), "Jan Oblak")

    assert (oblak["estado"], oblak["fidelidad"]) == ("sin_datos", None)


def test_el_contexto_del_juez_son_solo_los_insights_que_recibio_el_resumen():
    baja_confianza = "Pedri podria descansar segun un rumor de carrasco"
    otra_jornada = "Pedri fue el mejor de la jornada anterior segun carrasco"
    insights = [
        *INSIGHTS,
        insight("Pedri", baja_confianza, confianza=0.3),
        insight("Pedri", otra_jornada, jornada=JORNADA - 1),
    ]
    juez = JuezFalso()

    evaluar_fidelidad(recomendar(insights), juez=juez)

    (peticion,) = juez.peticiones_de_veredicto("Pedri vuelve a ser titular")
    candidatos = [TEXTO_PEDRI, TEXTO_PEDRI_RIVAL, TEXTO_RAPHINHA, baja_confianza, otra_jornada]
    assert [t for t in candidatos if t in peticion] == [TEXTO_PEDRI, TEXTO_PEDRI_RIVAL]


def test_el_juez_recibe_dos_peticiones_por_resumen_juzgado_y_ninguna_mas():
    juez = JuezFalso()

    evaluar(juez=juez)

    assert len(juez.peticiones) == 4


def test_un_resumen_del_que_el_juez_no_saca_afirmaciones_queda_sin_valor_y_fuera_de_la_media():
    juez = JuezFalso(veredictos={**VEREDICTOS, RESUMEN_RAPHINHA: []})

    resultado = evaluar(juez=juez)

    assert (detalle_de(resultado, "Raphinha")["fidelidad"], resultado["media"]) == (
        None,
        pytest.approx(0.75),
    )


def test_si_ningun_resumen_tiene_valor_la_media_es_nula():
    juez = JuezFalso(veredictos={RESUMEN_PEDRI: [], RESUMEN_RAPHINHA: []})

    assert evaluar(juez=juez)["media"] is None


def test_si_el_juez_falla_para_un_jugador_queda_con_el_error():
    juez = JuezFalso(fallos={RESUMEN_PEDRI: openai.OpenAIError("fallo de conexión")})

    pedri = detalle_de(evaluar(juez=juez), "Pedri")

    assert (pedri["estado"], "fallo de conexión" in pedri["error"]) == ("error", True)


def test_si_el_juez_falla_para_un_jugador_los_demas_se_juzgan_y_solo_ellos_hacen_la_media():
    juez = JuezFalso(fallos={RESUMEN_PEDRI: openai.OpenAIError("fallo de conexión")})

    assert evaluar(juez=juez)["media"] == pytest.approx(1.0)


def evaluar_con_oblak_sin_datos(juez):
    """Oblak recibe 2 insights, pero su resumen llega con `sin_datos`."""
    recomendacion = recomendar([*INSIGHTS, *INSIGHTS_OBLAK], acciones={"Jan Oblak": "sin_datos"})
    return evaluar_fidelidad(recomendacion, juez=juez)


def test_un_resumen_evitar_se_juzga_como_los_demas():
    juez = JuezFalso(
        veredictos={
            **VEREDICTOS,
            RESUMEN_RAPHINHA: [
                ("Raphinha encadena tres partidos", 1, "Lo dice el insight"),
                ("Raphinha esta lesionado", 0, "Ningun insight habla de lesion"),
            ],
        }
    )
    recomendacion = recomendar(acciones={"Raphinha": "evitar"})

    raphinha = detalle_de(evaluar_fidelidad(recomendacion, juez=juez), "Raphinha")

    assert raphinha["fidelidad"] == pytest.approx(0.5)


def test_un_resumen_sin_datos_con_insights_recibidos_no_se_juzga():
    resultado = evaluar_con_oblak_sin_datos(JuezFalso(veredictos=VEREDICTOS_CON_OBLAK))

    oblak = detalle_de(resultado, "Jan Oblak")
    assert (oblak["estado"], oblak["fidelidad"], oblak["no_respaldadas"]) == ("sin_datos", None, [])


def test_un_resumen_sin_datos_con_insights_recibidos_queda_fuera_de_la_media():
    resultado = evaluar_con_oblak_sin_datos(JuezFalso(veredictos=VEREDICTOS_CON_OBLAK))

    assert resultado["media"] == pytest.approx(0.875)


def test_el_juez_no_recibe_peticiones_por_un_resumen_sin_datos_con_insights_recibidos():
    juez = JuezFalso(veredictos=VEREDICTOS_CON_OBLAK)

    evaluar_con_oblak_sin_datos(juez)

    assert len(juez.peticiones) == 4
