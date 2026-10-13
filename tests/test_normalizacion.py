import json
import re
from types import SimpleNamespace

import openai

from fantasy_copilot.configuracion import cargar_configuracion
from fantasy_copilot.normalizacion import normalizar_insights

RAPHINHA = {
    "id": "411295",
    "nombre": "Raphinha",
    "posicion": "Extremo derecho",
    "equipo": "FC Barcelona",
}
LAMINE = {
    "id": "937958",
    "nombre": "Lamine Yamal",
    "posicion": "Extremo derecho",
    "equipo": "FC Barcelona",
}
GONZALO_MADRID = {
    "id": "111",
    "nombre": "Gonzalo García",
    "posicion": "Delantero centro",
    "equipo": "Real Madrid",
}
GONZALO_ESPANYOL = {
    "id": "222",
    "nombre": "Gonzalo García",
    "posicion": "Lateral izquierdo",
    "equipo": "RCD Espanyol",
}
OFICIALES = [RAPHINHA, LAMINE, GONZALO_MADRID, GONZALO_ESPANYOL]

RAPHINHA_RESUELTO = {
    "jugador": "Raphinha",
    "jugador_id": "411295",
    "equipo": "FC Barcelona",
    "posicion": "Extremo derecho",
    "nombre_transcrito": "Rafiña",
}

_ESCAPE_UNICODE = re.compile(r"\\+u([0-9a-fA-F]{4})")


def insight(jugador, quote="Es fijo esta jornada", **cambios):
    """Un insight con el formato de `extraer_insights`."""
    return {
        "jugador": jugador,
        "accion": "alinear",
        "confianza": 0.9,
        "razon_corta": "En racha y con rival asequible",
        "criterios": {
            "forma_actual": "Tres goles en dos partidos",
            "dinamica_equipo": None,
            "calendario_rival": None,
            "prediccion_puntos": None,
        },
        "quote_original": quote,
        "fuente": {
            "canal": "carrasco",
            "video_id": "AT_UTIeNmtI",
            "titulo": "Jornada 12",
            "fecha": "2026-11-05",
            "jornada": 12,
            "temporada": 2026,
            "url": "https://youtube.com/watch?v=AT_UTIeNmtI",
            "chunk": None,
        },
        **cambios,
    }


def elige(jugador_id):
    mensaje = SimpleNamespace(
        role="assistant", content=json.dumps({"jugador_id": jugador_id}), refusal=None
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=mensaje, finish_reason="stop")])


class OpenAIFalso:
    """Hace de cliente de OpenAI: responde a `chat.completions.create`.

    `reglas` es una lista de pares (marca, respuesta). Se aplica la primera regla cuya marca
    aparece en los mensajes de la petición. Si la respuesta es una excepción, se lanza.
    """

    def __init__(self, reglas):
        self.reglas = reglas
        self.peticiones = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._crear))

    def _crear(self, **peticion):
        enviado = json.dumps(peticion.get("messages"), ensure_ascii=False, default=str)
        enviado = _ESCAPE_UNICODE.sub(lambda escape: chr(int(escape.group(1), 16)), enviado)
        self.peticiones.append({"modelo": peticion.get("model"), "enviado": enviado})
        respuesta = next((r for marca, r in self.reglas if marca in enviado), None)
        if respuesta is None:
            raise AssertionError("Petición inesperada: ninguna regla coincide con su texto")
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta


def normalizar(insights, datos_dir, cliente, jornada=12, modelo="modelo-de-prueba"):
    return normalizar_insights(
        insights,
        OFICIALES,
        jornada=jornada,
        temporada=2026,
        datos_dir=datos_dir,
        modelo=modelo,
        cliente=cliente,
    )


def leer(datos_dir, jornada, nombre):
    fichero = datos_dir / "2026" / f"j{jornada:02d}" / nombre
    return json.loads(fichero.read_text(encoding="utf-8"))


def test_un_nombre_resuelto_deja_el_insight_con_los_datos_del_jugador_oficial(tmp_path):
    oido = insight("Rafiña", "Rafiña es fijo esta jornada")
    openai_falso = OpenAIFalso([("Rafiña", elige("411295"))])

    resultado = normalizar([oido], tmp_path, openai_falso)

    assert resultado["insights"] == [{**oido, **RAPHINHA_RESUELTO}]


def test_un_nombre_sin_jugador_no_aparece_en_los_insights(tmp_path):
    openai_falso = OpenAIFalso([("75 kg para Moreno", elige(None)), ("Rafiña", elige("411295"))])

    resultado = normalizar(
        [insight("75 kg para Moreno"), insight("Rafiña")], tmp_path, openai_falso
    )

    assert [i["nombre_transcrito"] for i in resultado["insights"]] == ["Rafiña"]


def test_un_nombre_sin_jugador_queda_eliminado_con_su_cita_y_sin_coincidencia(tmp_path):
    openai_falso = OpenAIFalso([("75 kg para Moreno", elige(None))])

    resultado = normalizar(
        [insight("75 kg para Moreno", "Pesa 75 kg para Moreno")], tmp_path, openai_falso
    )

    assert resultado["eliminados"] == [
        {
            "nombre": "75 kg para Moreno",
            "quote": "Pesa 75 kg para Moreno",
            "motivo": "sin_coincidencia",
        }
    ]


def test_un_identificador_que_no_esta_en_la_lista_oficial_elimina_el_nombre(tmp_path):
    openai_falso = OpenAIFalso([("Rafiña", elige("999999"))])

    resultado = normalizar([insight("Rafiña", "Rafiña es fijo")], tmp_path, openai_falso)

    assert resultado == {
        "insights": [],
        "eliminados": [
            {"nombre": "Rafiña", "quote": "Rafiña es fijo", "motivo": "sin_coincidencia"}
        ],
    }


def test_la_decision_de_un_nombre_se_aplica_a_todos_sus_insights(tmp_path):
    openai_falso = OpenAIFalso([("Rafiña", elige("411295"))])
    oidos = [insight("Rafiña", f"Cita {n}") for n in (1, 2, 3)]

    resultado = normalizar(oidos, tmp_path, openai_falso)

    assert [(i["jugador_id"], i["quote_original"]) for i in resultado["insights"]] == [
        ("411295", "Cita 1"),
        ("411295", "Cita 2"),
        ("411295", "Cita 3"),
    ]


def test_un_nombre_repetido_en_tres_insights_se_consulta_una_sola_vez(tmp_path):
    openai_falso = OpenAIFalso([("Rafiña", elige("411295"))])
    oidos = [insight("Rafiña", f"Cita {n}") for n in (1, 2, 3)]

    normalizar(oidos, tmp_path, openai_falso)

    assert len(openai_falso.peticiones) == 1


def test_la_cita_de_un_nombre_eliminado_es_la_del_primer_insight_en_que_aparece(tmp_path):
    openai_falso = OpenAIFalso([("75 kg para Moreno", elige(None))])
    oidos = [
        insight("75 kg para Moreno", "Primera cita"),
        insight("75 kg para Moreno", "Segunda cita"),
    ]

    resultado = normalizar(oidos, tmp_path, openai_falso)

    assert resultado["eliminados"] == [
        {"nombre": "75 kg para Moreno", "quote": "Primera cita", "motivo": "sin_coincidencia"}
    ]


def test_entre_dos_jugadores_con_el_mismo_nombre_se_asigna_el_del_identificador_elegido(tmp_path):
    openai_falso = OpenAIFalso([("Gonzalo García", elige("222"))])

    resultado = normalizar([insight("Gonzalo García")], tmp_path, openai_falso)

    assert [(i["jugador_id"], i["equipo"], i["posicion"]) for i in resultado["insights"]] == [
        ("222", "RCD Espanyol", "Lateral izquierdo")
    ]


def test_el_modelo_recibe_los_identificadores_de_los_dos_jugadores_con_el_mismo_nombre(tmp_path):
    openai_falso = OpenAIFalso([("Gonzalo García", elige("222"))])

    normalizar([insight("Gonzalo García")], tmp_path, openai_falso)

    enviado = openai_falso.peticiones[0]["enviado"]
    assert "111" in enviado and "222" in enviado


def test_el_modelo_recibe_el_equipo_de_cada_candidato(tmp_path):
    openai_falso = OpenAIFalso([("Gonzalo García", elige("222"))])

    normalizar([insight("Gonzalo García")], tmp_path, openai_falso)

    enviado = openai_falso.peticiones[0]["enviado"]
    assert "Real Madrid" in enviado and "RCD Espanyol" in enviado


def test_un_error_de_la_api_elimina_ese_nombre_con_error_api(tmp_path):
    openai_falso = OpenAIFalso([("la mine", openai.OpenAIError("fallo de conexión"))])

    resultado = normalizar([insight("la mine", "la mine es duda")], tmp_path, openai_falso)

    assert resultado["eliminados"] == [
        {"nombre": "la mine", "quote": "la mine es duda", "motivo": "error_api"}
    ]


def test_tras_un_error_de_la_api_los_demas_nombres_se_siguen_resolviendo(tmp_path):
    openai_falso = OpenAIFalso(
        [("la mine", openai.OpenAIError("fallo de conexión")), ("Rafiña", elige("411295"))]
    )

    resultado = normalizar([insight("la mine"), insight("Rafiña")], tmp_path, openai_falso)

    assert [i["jugador_id"] for i in resultado["insights"]] == ["411295"]


def test_los_insights_resueltos_se_guardan_en_la_carpeta_de_la_jornada(tmp_path):
    oido = insight("Rafiña", "Rafiña es fijo esta jornada")
    openai_falso = OpenAIFalso([("Rafiña", elige("411295"))])

    normalizar([oido], tmp_path, openai_falso, jornada=3)

    assert leer(tmp_path, 3, "insights_normalizados.json") == [{**oido, **RAPHINHA_RESUELTO}]


def test_los_nombres_eliminados_se_guardan_en_la_carpeta_de_la_jornada(tmp_path):
    openai_falso = OpenAIFalso([("75 kg para Moreno", elige(None))])

    normalizar(
        [insight("75 kg para Moreno", "Pesa 75 kg para Moreno")], tmp_path, openai_falso, jornada=3
    )

    assert leer(tmp_path, 3, "nombres_eliminados.json") == [
        {
            "nombre": "75 kg para Moreno",
            "quote": "Pesa 75 kg para Moreno",
            "motivo": "sin_coincidencia",
        }
    ]


def test_las_peticiones_usan_el_modelo_de_normalizacion_de_la_configuracion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FANTASY_MODELO_NORMALIZACION", "modelo-y")
    openai_falso = OpenAIFalso([("Rafiña", elige("411295"))])

    normalizar(
        [insight("Rafiña")],
        tmp_path,
        openai_falso,
        modelo=cargar_configuracion().modelo_normalizacion,
    )

    assert [peticion["modelo"] for peticion in openai_falso.peticiones] == ["modelo-y"]
