import json
import re
from types import SimpleNamespace

import openai
import pytest

from fantasy_copilot.agente import ErrorRecomendacion, recomendar_once
from fantasy_copilot.configuracion import cargar_configuracion

JORNADA = 12
TEMPORADA = 2026

PLANTILLA = [
    {"nombre": nombre, "jugador_id": jugador_id, "equipo": equipo, "posicion": posicion}
    for nombre, jugador_id, equipo, posicion in [
        ("Jan Oblak", "121483", "Atlético de Madrid", "portero"),
        ("Jules Koundé", "411975", "FC Barcelona", "defensa"),
        ("Pau Cubarsí", "937945", "FC Barcelona", "defensa"),
        ("Dani Vivian", "462250", "Athletic Club", "defensa"),
        ("Miguel Gutiérrez", "683723", "Girona FC", "defensa"),
        ("Pedri", "683840", "FC Barcelona", "centrocampista"),
        ("Federico Valverde", "369081", "Real Madrid", "centrocampista"),
        ("Raphinha", "411295", "FC Barcelona", "delantero"),
        ("Lamine Yamal", "937958", "FC Barcelona", "delantero"),
        ("Julián Álvarez", "576024", "Atlético de Madrid", "delantero"),
        ("Nico Williams", "709187", "Athletic Club", "delantero"),
        ("Augusto Batalla", "264335", "Rayo Vallecano", "portero"),
    ]
]
NOMBRES = [jugador["nombre"] for jugador in PLANTILLA]
POSICION = {jugador["nombre"]: jugador["posicion"] for jugador in PLANTILLA}

RESUMEN_RAPHINHA = {
    "jugador": "Raphinha",
    "posicion": "delantero",
    "accion_recomendada": "alinear",
    "confianza_media": 0.85,
    "resumen": "En racha y con rival asequible; dos analistas lo dan por fijo.",
    "fuentes": ["carrasco - Jornada 12"],
}

_ESCAPE_UNICODE = re.compile(r"\\+u([0-9a-fA-F]{4})")


def insight(jugador, confianza=0.9, canal="carrasco"):
    """Un resultado de `Indice.consultar` sobre `jugador`."""
    posicion = POSICION.get(jugador, "")
    return {
        "texto": (
            f"{jugador}, {posicion}. Recomendación: alinear. "
            f"Razón: En racha y con rival asequible. Fuente: {canal}, Jornada {JORNADA}."
        ),
        "metadatos": {
            "jugador": jugador,
            "jugador_id": "0",
            "equipo": "FC Barcelona",
            "posicion": posicion,
            "accion": "alinear",
            "confianza": confianza,
            "canal": canal,
            "fecha": "2026-11-05",
            "jornada": JORNADA,
            "temporada": TEMPORADA,
            "url": f"https://youtube.com/watch?v={canal}",
        },
        "distancia": 0.1,
    }


class IndiceFalso:
    """Hace de índice: devuelve los insights cuyo jugador aparece en la pregunta."""

    def __init__(self, insights=()):
        self.insights = list(insights)

    def consultar(self, pregunta, *, jornada, temporada, top_k=10):
        encontrados = [
            i
            for i in self.insights
            if i["metadatos"]["jugador"] in pregunta
            and i["metadatos"]["jornada"] == jornada
            and i["metadatos"]["temporada"] == temporada
        ]
        return encontrados[:top_k]


def indice_con_todos():
    return IndiceFalso([insight(nombre) for nombre in NOMBRES])


def _respuesta(contenido):
    mensaje = SimpleNamespace(
        role="assistant",
        content=json.dumps(contenido, ensure_ascii=False),
        refusal=None,
        tool_calls=None,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=mensaje, finish_reason="stop")])


def resumen_del_modelo(nombre):
    if nombre == "Raphinha":
        return _respuesta(RESUMEN_RAPHINHA)
    return _respuesta(
        {
            "jugador": nombre,
            "posicion": POSICION[nombre],
            "accion_recomendada": "alinear",
            "confianza_media": 0.7,
            "resumen": "Titular habitual.",
            "fuentes": ["carrasco - Jornada 12"],
        }
    )


def recomendacion(nombres, advertencias=("Aviso del modelo",)):
    return {
        "once_recomendado": [
            {
                "jugador": nombre,
                "posicion": POSICION.get(nombre, "delantero"),
                "confianza": 0.8,
                "justificacion": "En racha y con rival asequible",
            }
            for nombre in nombres
        ],
        "advertencias": list(advertencias),
        "resumen": "Once ofensivo aprovechando el calendario.",
    }


ONCE_VALIDO = recomendacion(NOMBRES[:11])


def error_api():
    return openai.OpenAIError("fallo de conexión")


class OpenAIFalso:
    """Hace de cliente de OpenAI: responde a `chat.completions.create`.

    Una petición que nombra a más de tres jugadores de la plantilla es la final; si nombra a uno
    solo, es el resumen de ese jugador. `resumenes` fija la respuesta por jugador (por defecto,
    `resumen_del_modelo`) y `final` la de la petición final. Una respuesta puede ser una
    excepción, que se lanza, o una lista, de la que se consume un elemento por petición.
    """

    def __init__(self, final=None, resumenes=None):
        self.final = _respuesta(ONCE_VALIDO) if final is None else final
        self.resumenes = dict(resumenes or {})
        self.peticiones = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._crear))

    def _crear(self, **peticion):
        enviado = json.dumps(peticion.get("messages"), ensure_ascii=False, default=str)
        enviado = _ESCAPE_UNICODE.sub(lambda escape: chr(int(escape.group(1), 16)), enviado)
        nombrados = [nombre for nombre in NOMBRES if nombre in enviado]
        if len(nombrados) > 3:
            tipo = "final"
        elif len(nombrados) == 1:
            tipo = nombrados[0]
        else:
            raise AssertionError(f"Petición inesperada: nombra a {nombrados}")
        self.peticiones.append(
            {
                "modelo": peticion.get("model"),
                "enviado": enviado,
                "tipo": tipo,
                "nombrados": nombrados,
            }
        )
        if tipo == "final":
            respuesta = self.final
        else:
            respuesta = self.resumenes.get(tipo) or resumen_del_modelo(tipo)
        if isinstance(respuesta, list):
            respuesta = respuesta.pop(0) if len(respuesta) > 1 else respuesta[0]
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta

    def de_tipo(self, tipo):
        return [peticion for peticion in self.peticiones if peticion["tipo"] == tipo]


def recomendar(indice, cliente, modelo="modelo-de-prueba"):
    return recomendar_once(
        PLANTILLA,
        jornada=JORNADA,
        temporada=TEMPORADA,
        indice=indice,
        modelo=modelo,
        cliente=cliente,
        esperar=lambda segundos: None,
    )


def resumen_de(resultado, nombre):
    return next(r for r in resultado["resumenes"] if r["jugador"] == nombre)


def test_un_jugador_sin_insights_de_confianza_suficiente_queda_sin_datos_con_confianza_cero():
    indice = IndiceFalso([insight("Raphinha", confianza=0.3)])

    resultado = recomendar(indice, OpenAIFalso())

    resumen = resumen_de(resultado, "Raphinha")
    assert (resumen["accion_recomendada"], resumen["confianza_media"]) == ("sin_datos", 0)


def test_no_se_consulta_al_modelo_por_los_jugadores_sin_insights_utilizables():
    indice = IndiceFalso([insight("Raphinha", confianza=0.3)])
    openai_falso = OpenAIFalso()

    recomendar(indice, openai_falso)

    assert [peticion["tipo"] for peticion in openai_falso.peticiones] == ["final"]


def test_el_resumen_de_un_jugador_con_insights_es_el_que_devuelve_el_modelo():
    indice = IndiceFalso([insight("Raphinha")])

    resultado = recomendar(indice, OpenAIFalso())

    assert resumen_de(resultado, "Raphinha") == RESUMEN_RAPHINHA


def test_el_resumen_se_pide_solo_con_los_insights_de_confianza_0_4_o_mas():
    indice = IndiceFalso(
        [
            insight("Raphinha", confianza=0.3, canal="canal_bajo"),
            insight("Raphinha", confianza=0.4, canal="canal_limite"),
            insight("Raphinha", confianza=0.8, canal="canal_alto"),
        ]
    )
    openai_falso = OpenAIFalso()

    recomendar(indice, openai_falso)

    (peticion,) = openai_falso.de_tipo("Raphinha")
    canales = ["canal_bajo", "canal_limite", "canal_alto"]
    assert [canal for canal in canales if canal in peticion["enviado"]] == [
        "canal_limite",
        "canal_alto",
    ]


def test_una_unica_peticion_final_recibe_a_todos_los_jugadores_incluidos_los_sin_datos():
    indice = IndiceFalso([insight("Raphinha")])
    openai_falso = OpenAIFalso()

    recomendar(indice, openai_falso)

    finales = openai_falso.de_tipo("final")
    assert [sorted(peticion["nombrados"]) for peticion in finales] == [sorted(NOMBRES)]


def test_la_recomendacion_de_un_once_valido_es_la_que_devuelve_el_modelo():
    resultado = recomendar(indice_con_todos(), OpenAIFalso(final=_respuesta(ONCE_VALIDO)))

    assert resultado["recomendacion"] == ONCE_VALIDO


def test_un_once_no_valido_se_devuelve_igualmente():
    diez = recomendacion(NOMBRES[:10])

    resultado = recomendar(indice_con_todos(), OpenAIFalso(final=_respuesta(diez)))

    assert resultado["recomendacion"]["once_recomendado"] == diez["once_recomendado"]


def test_un_once_de_diez_jugadores_lleva_un_aviso_con_el_numero_recibido():
    diez = recomendacion(NOMBRES[:10])

    resultado = recomendar(indice_con_todos(), OpenAIFalso(final=_respuesta(diez)))

    assert any("10" in aviso for aviso in resultado["recomendacion"]["advertencias"])


def test_un_once_con_un_jugador_repetido_lleva_un_aviso():
    repetido = recomendacion([*NOMBRES[:10], NOMBRES[0]], advertencias=())

    resultado = recomendar(indice_con_todos(), OpenAIFalso(final=_respuesta(repetido)))

    assert resultado["recomendacion"]["advertencias"] != []


def test_un_once_con_un_jugador_fuera_de_la_plantilla_lleva_un_aviso_con_su_nombre():
    con_intruso = recomendacion([*NOMBRES[:10], "Kylian Mbappé"])

    resultado = recomendar(indice_con_todos(), OpenAIFalso(final=_respuesta(con_intruso)))

    assert any("Kylian Mbappé" in aviso for aviso in resultado["recomendacion"]["advertencias"])


def test_un_resumen_que_falla_dos_veces_y_luego_responde_es_el_del_modelo():
    indice = IndiceFalso([insight("Raphinha")])
    openai_falso = OpenAIFalso(
        resumenes={"Raphinha": [error_api(), error_api(), _respuesta(RESUMEN_RAPHINHA)]}
    )

    resultado = recomendar(indice, openai_falso)

    assert resumen_de(resultado, "Raphinha") == RESUMEN_RAPHINHA


def test_un_resumen_que_falla_tres_veces_deja_al_jugador_sin_datos():
    indice = IndiceFalso([insight("Raphinha")])
    openai_falso = OpenAIFalso(
        resumenes={
            "Raphinha": [error_api(), error_api(), error_api(), _respuesta(RESUMEN_RAPHINHA)]
        }
    )

    resultado = recomendar(indice, openai_falso)

    assert resumen_de(resultado, "Raphinha")["accion_recomendada"] == "sin_datos"


def test_un_resumen_fallido_deja_un_aviso_con_el_nombre_del_jugador():
    indice = IndiceFalso([insight("Raphinha")])
    openai_falso = OpenAIFalso(
        final=_respuesta(recomendacion(NOMBRES[:11], advertencias=())),
        resumenes={"Raphinha": error_api()},
    )

    resultado = recomendar(indice, openai_falso)

    assert any("Raphinha" in aviso for aviso in resultado["recomendacion"]["advertencias"])


def test_tras_fallar_el_resumen_de_un_jugador_los_demas_tienen_el_del_modelo():
    indice = IndiceFalso([insight("Raphinha"), insight("Lamine Yamal")])
    openai_falso = OpenAIFalso(resumenes={"Raphinha": error_api()})

    resultado = recomendar(indice, openai_falso)

    assert resumen_de(resultado, "Lamine Yamal")["accion_recomendada"] == "alinear"


def test_si_falla_la_peticion_final_se_produce_un_error():
    indice = IndiceFalso([insight("Raphinha")])
    openai_falso = OpenAIFalso(final=error_api())

    with pytest.raises(ErrorRecomendacion):
        recomendar(indice, openai_falso)


def test_las_peticiones_usan_el_modelo_del_agente_de_la_configuracion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FANTASY_MODELO_AGENTE", "modelo-z")
    indice = IndiceFalso([insight("Raphinha")])
    openai_falso = OpenAIFalso()

    recomendar(indice, openai_falso, modelo=cargar_configuracion().modelo_agente)

    assert [peticion["modelo"] for peticion in openai_falso.peticiones] == ["modelo-z", "modelo-z"]
