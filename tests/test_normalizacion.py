import json
import re
from types import SimpleNamespace

import openai
import pytest

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
        role="assistant",
        content=json.dumps({"jugador_id": jugador_id}),
        refusal=None,
        tool_calls=None,
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
        if isinstance(respuesta, BaseException):
            raise respuesta
        return respuesta


def normalizar(
    insights, datos_dir, cliente, jornada=12, modelo="modelo-de-prueba", oficiales=OFICIALES
):
    return normalizar_insights(
        insights,
        oficiales,
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


def oficial(identificador, nombre, posicion, *equipos):
    """Un jugador con el formato de la lista oficial: `equipo` es el primero de `equipos`."""
    return {
        "id": identificador,
        "nombre": nombre,
        "posicion": posicion,
        "equipo": equipos[0],
        "equipos": list(equipos),
    }


PEPELU = oficial("500101", "Pepelu", "Pivote", "Valencia CF")
GERARD_MORENO = oficial("500202", "Gerard Moreno", "Delantero centro", "Villarreal CF")
MAYORAL = oficial("500303", "Borja Mayoral", "Delantero centro", "Getafe CF", "Real Betis Balompié")
DJENE = oficial("500404", "Djené", "Defensa central", "Getafe CF")
ISCO = oficial("500505", "Isco", "Mediapunta", "Real Betis Balompié")
CON_EQUIPOS = [
    oficial("411295", "Raphinha", "Extremo derecho", "FC Barcelona"),
    PEPELU,
    GERARD_MORENO,
    MAYORAL,
    DJENE,
    ISCO,
]

MAXIMO_DE_PETICIONES = 10


def pide(*llamadas):
    """Respuesta del modelo que pide herramientas: cada llamada es (id, herramienta, argumentos).

    Si `argumentos` es un texto se envía tal cual, para poder mandar JSON mal formado.
    """
    tool_calls = [
        SimpleNamespace(
            id=identificador,
            type="function",
            function=SimpleNamespace(
                name=herramienta,
                arguments=argumentos
                if isinstance(argumentos, str)
                else json.dumps(argumentos, ensure_ascii=False),
            ),
        )
        for identificador, herramienta, argumentos in llamadas
    ]
    mensaje = SimpleNamespace(role="assistant", content=None, refusal=None, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=mensaje, finish_reason="tool_calls")])


def ofrece_herramientas(peticion):
    return bool(peticion.get("tools")) and peticion.get("tool_choice") != "none"


class OpenAIConGuion:
    """Hace de cliente de OpenAI y guarda cada petición tal como se envió.

    `responder` recibe la petición y devuelve la respuesta. Corta con un error si la
    conversación no termina.
    """

    def __init__(self, responder):
        self.responder = responder
        self.peticiones = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._crear))

    def _crear(self, **peticion):
        if len(self.peticiones) >= MAXIMO_DE_PETICIONES:
            raise AssertionError("La resolución del nombre no termina")
        self.peticiones.append({**peticion, "messages": list(peticion["messages"])})
        return self.responder(peticion)

    def resultados(self):
        """Lo que el modelo ha recibido de cada herramienta, por identificador de llamada."""
        recibido = {}
        for peticion in self.peticiones:
            for mensaje in peticion["messages"]:
                if isinstance(mensaje, dict) and mensaje.get("role") == "tool":
                    recibido[mensaje["tool_call_id"]] = mensaje["content"]
        return recibido


def en_orden(*respuestas):
    """Devuelve las respuestas una a una y repite la última."""
    pendientes = list(respuestas)

    def responder(peticion):
        return pendientes.pop(0) if len(pendientes) > 1 else pendientes[0]

    return responder


def pide_equipo_mientras_se_ofrezcan_herramientas(final):
    turnos = iter(range(MAXIMO_DE_PETICIONES + 1))

    def responder(peticion):
        if not ofrece_herramientas(peticion):
            return final
        return pide((f"llamada_{next(turnos)}", "buscar_por_equipo", {"equipo": "Getafe"}))

    return responder


def resultado_de(herramienta, argumentos, tmp_path, oficiales=CON_EQUIPOS):
    """Lo que recibe el modelo al pedir una sola herramienta mientras resuelve un nombre."""
    openai_falso = OpenAIConGuion(
        en_orden(pide(("llamada_1", herramienta, argumentos)), elige(None))
    )
    normalizar([insight("Rafiña")], tmp_path, openai_falso, oficiales=oficiales)
    return openai_falso.resultados()["llamada_1"]


def mencionados(texto, campo, oficiales=CON_EQUIPOS):
    return {jugador[campo] for jugador in oficiales if jugador[campo] in texto}


def test_si_el_modelo_no_pide_herramientas_el_nombre_se_resuelve_con_una_sola_peticion(tmp_path):
    openai_falso = OpenAIConGuion(en_orden(elige("411295")))

    resultado = normalizar([insight("Rafiña")], tmp_path, openai_falso, oficiales=CON_EQUIPOS)

    assert [i["jugador_id"] for i in resultado["insights"]] == ["411295"]
    assert len(openai_falso.peticiones) == 1


def test_al_modelo_se_le_ofrecen_las_dos_herramientas_con_sus_argumentos(tmp_path):
    openai_falso = OpenAIConGuion(en_orden(elige("411295")))

    normalizar([insight("Rafiña")], tmp_path, openai_falso, oficiales=CON_EQUIPOS)

    ofrecidas = {
        herramienta["function"]["name"]: sorted(herramienta["function"]["parameters"]["properties"])
        for herramienta in openai_falso.peticiones[0].get("tools") or []
    }
    assert ofrecidas == {"buscar_en_lista_oficial": ["nombre"], "buscar_por_equipo": ["equipo"]}


def test_buscar_en_lista_oficial_devuelve_el_jugador_con_identificador_nombre_y_equipo(tmp_path):
    recibido = resultado_de("buscar_en_lista_oficial", {"nombre": "Pepelu"}, tmp_path)

    assert mencionados(recibido, "id") == {"500101"}
    assert "Pepelu" in recibido and "Valencia CF" in recibido


def test_buscar_en_lista_oficial_no_devuelve_ningun_jugador_si_el_nombre_no_se_parece(tmp_path):
    recibido = resultado_de("buscar_en_lista_oficial", {"nombre": "75 kg para Moreno"}, tmp_path)

    assert mencionados(recibido, "id") == set()
    assert mencionados(recibido, "nombre") == set()


def test_buscar_por_equipo_devuelve_identificador_y_nombre_de_los_jugadores_del_equipo(tmp_path):
    recibido = resultado_de("buscar_por_equipo", {"equipo": "Getafe"}, tmp_path)

    assert mencionados(recibido, "id") == {"500303", "500404"}
    assert mencionados(recibido, "nombre") == {"Borja Mayoral", "Djené"}


def test_buscar_por_equipo_incluye_al_jugador_que_figura_en_ese_equipo_sin_ser_el_primero(
    tmp_path,
):
    recibido = resultado_de("buscar_por_equipo", {"equipo": "Betis"}, tmp_path)

    assert mencionados(recibido, "id") == {"500303", "500505"}


def test_buscar_por_equipo_no_devuelve_ningun_jugador_si_el_equipo_no_se_parece(tmp_path):
    recibido = resultado_de("buscar_por_equipo", {"equipo": "Xyzzy"}, tmp_path)

    assert mencionados(recibido, "id") == set()
    assert mencionados(recibido, "nombre") == set()


def test_tras_tres_llamadas_la_siguiente_peticion_ya_no_ofrece_herramientas(tmp_path):
    openai_falso = OpenAIConGuion(pide_equipo_mientras_se_ofrezcan_herramientas(elige("411295")))

    normalizar([insight("Rafiña")], tmp_path, openai_falso, oficiales=CON_EQUIPOS)

    assert [ofrece_herramientas(p) for p in openai_falso.peticiones] == [True, True, True, False]


def test_agotadas_las_llamadas_el_nombre_se_resuelve_con_la_respuesta_final(tmp_path):
    openai_falso = OpenAIConGuion(pide_equipo_mientras_se_ofrezcan_herramientas(elige("411295")))

    resultado = normalizar([insight("Rafiña")], tmp_path, openai_falso, oficiales=CON_EQUIPOS)

    assert [i["jugador_id"] for i in resultado["insights"]] == ["411295"]
    assert len(openai_falso.resultados()) == 3


def dos_turnos_de_dos_llamadas():
    return en_orden(
        pide(
            ("llamada_1", "buscar_en_lista_oficial", {"nombre": "Raphinha"}),
            ("llamada_2", "buscar_en_lista_oficial", {"nombre": "Gerard Moreno"}),
        ),
        pide(
            ("llamada_3", "buscar_en_lista_oficial", {"nombre": "Pepelu"}),
            ("llamada_4", "buscar_por_equipo", {"equipo": "Getafe"}),
        ),
        elige("411295"),
    )


def test_con_dos_llamadas_por_turno_la_tercera_se_ejecuta(tmp_path):
    openai_falso = OpenAIConGuion(dos_turnos_de_dos_llamadas())

    normalizar([insight("Rafiña")], tmp_path, openai_falso, oficiales=CON_EQUIPOS)

    assert mencionados(openai_falso.resultados()["llamada_3"], "id") == {"500101"}


def test_con_dos_llamadas_por_turno_la_cuarta_no_se_ejecuta(tmp_path):
    openai_falso = OpenAIConGuion(dos_turnos_de_dos_llamadas())

    normalizar([insight("Rafiña")], tmp_path, openai_falso, oficiales=CON_EQUIPOS)

    recibido = openai_falso.resultados()
    assert {"llamada_1", "llamada_2", "llamada_3"} <= set(recibido)
    assert mencionados(" ".join(recibido.values()), "id") == {"411295", "500202", "500101"}


GRIEZMANN = oficial("600101", "Antoine Griezmann", "Delantero centro", "Atlético de Madrid")
KOKE = oficial("600202", "Koke", "Mediocentro", "Atlético de Madrid")
VINICIUS = oficial("600303", "Vinicius Junior", "Extremo izquierdo", "Real Madrid CF")
VALVERDE = oficial("600404", "Federico Valverde", "Mediocentro", "Real Madrid CF")
CON_MADRILENOS = [*CON_EQUIPOS, GRIEZMANN, KOKE, VINICIUS, VALVERDE]

ARGUMENTOS_MAL_FORMADOS = pytest.mark.parametrize(
    "argumentos",
    ['{"equipo":', {"nombre": "Getafe"}],
    ids=["json_no_valido", "falta_el_argumento_equipo"],
)


def pide_mal_formado_mientras_se_ofrezcan_herramientas(final):
    turnos = iter(range(MAXIMO_DE_PETICIONES + 1))

    def responder(peticion):
        if not ofrece_herramientas(peticion):
            return final
        return pide((f"llamada_{next(turnos)}", "buscar_por_equipo", '{"equipo":'))

    return responder


@pytest.mark.parametrize("nombre", ["pepelu", "PEPELU"])
def test_buscar_en_lista_oficial_encuentra_al_jugador_sin_distinguir_mayusculas(nombre, tmp_path):
    recibido = resultado_de("buscar_en_lista_oficial", {"nombre": nombre}, tmp_path)

    assert mencionados(recibido, "id") == {"500101"}
    assert "Pepelu" in recibido and "Valencia CF" in recibido


@pytest.mark.parametrize("equipo", ["GETAFE", "getafe"])
def test_buscar_por_equipo_encuentra_el_equipo_sin_distinguir_mayusculas(equipo, tmp_path):
    recibido = resultado_de("buscar_por_equipo", {"equipo": equipo}, tmp_path)

    assert mencionados(recibido, "id") == {"500303", "500404"}
    assert mencionados(recibido, "nombre") == {"Borja Mayoral", "Djené"}


def test_buscar_por_equipo_devuelve_solo_el_equipo_mas_parecido_si_no_hay_empate(tmp_path):
    recibido = resultado_de(
        "buscar_por_equipo", {"equipo": "Real Madrid"}, tmp_path, oficiales=CON_MADRILENOS
    )

    assert mencionados(recibido, "id", CON_MADRILENOS) == {"600303", "600404"}


def test_buscar_por_equipo_devuelve_los_jugadores_de_todos_los_equipos_empatados(tmp_path):
    recibido = resultado_de(
        "buscar_por_equipo", {"equipo": "Madrid"}, tmp_path, oficiales=CON_MADRILENOS
    )

    assert mencionados(recibido, "id", CON_MADRILENOS) == {"600101", "600202", "600303", "600404"}


def test_buscar_por_equipo_dice_el_equipo_de_los_jugadores_cuando_hay_empate(tmp_path):
    recibido = resultado_de(
        "buscar_por_equipo", {"equipo": "Madrid"}, tmp_path, oficiales=CON_MADRILENOS
    )

    assert mencionados(recibido, "equipo", CON_MADRILENOS) == {
        "Atlético de Madrid",
        "Real Madrid CF",
    }


@ARGUMENTOS_MAL_FORMADOS
def test_tras_una_llamada_con_argumentos_mal_formados_el_nombre_se_resuelve_con_la_respuesta_final(
    argumentos, tmp_path
):
    openai_falso = OpenAIConGuion(
        en_orden(pide(("llamada_1", "buscar_por_equipo", argumentos)), elige("411295"))
    )

    resultado = normalizar([insight("Rafiña")], tmp_path, openai_falso, oficiales=CON_EQUIPOS)

    assert [i["jugador_id"] for i in resultado["insights"]] == ["411295"]
    assert resultado["eliminados"] == []


@ARGUMENTOS_MAL_FORMADOS
def test_una_llamada_con_argumentos_mal_formados_recibe_un_resultado_sin_jugadores(
    argumentos, tmp_path
):
    openai_falso = OpenAIConGuion(
        en_orden(pide(("llamada_1", "buscar_por_equipo", argumentos)), elige("411295"))
    )

    normalizar([insight("Rafiña")], tmp_path, openai_falso, oficiales=CON_EQUIPOS)

    recibido = openai_falso.resultados()
    assert set(recibido) == {"llamada_1"}
    assert recibido["llamada_1"].strip()
    assert mencionados(recibido["llamada_1"], "id") == set()
    assert mencionados(recibido["llamada_1"], "nombre") == set()


def test_tres_llamadas_con_argumentos_mal_formados_agotan_el_tope_de_herramientas(tmp_path):
    openai_falso = OpenAIConGuion(
        pide_mal_formado_mientras_se_ofrezcan_herramientas(elige("411295"))
    )

    normalizar([insight("Rafiña")], tmp_path, openai_falso, oficiales=CON_EQUIPOS)

    assert [ofrece_herramientas(p) for p in openai_falso.peticiones] == [True, True, True, False]


FICHEROS = ("insights_normalizados.json", "nombres_eliminados.json")
DECISIONES_PRIMERA = [("75 kg para Moreno", elige(None)), ("Rafiña", elige("411295"))]
# Si se volviera a consultar, estas respuestas cambiarían el resultado de la primera ejecución.
OTRAS_DECISIONES = [("75 kg para Moreno", elige("411295")), ("Rafiña", elige("937958"))]


def rafina(quote="Rafiña es fijo esta jornada"):
    return insight("Rafiña", quote)


def kilos():
    return insight("75 kg para Moreno", "Pesa 75 kg para Moreno")


def de_otro_video(oido):
    fuente = {
        **oido["fuente"],
        "video_id": "OTRO_VIDEO1",
        "url": "https://youtube.com/watch?v=OTRO_VIDEO1",
    }
    return {**oido, "fuente": fuente}


def primera_ejecucion(datos_dir, insights, reglas=DECISIONES_PRIMERA):
    return normalizar(insights, datos_dir, OpenAIFalso(reglas))


def test_una_segunda_ejecucion_con_los_mismos_insights_no_hace_ninguna_peticion(tmp_path):
    primera_ejecucion(tmp_path, [kilos(), rafina()])
    openai_falso = OpenAIFalso(OTRAS_DECISIONES)

    normalizar([kilos(), rafina()], tmp_path, openai_falso)

    assert openai_falso.peticiones == []


def test_una_segunda_ejecucion_con_los_mismos_insights_devuelve_el_mismo_resultado(tmp_path):
    primero = primera_ejecucion(tmp_path, [kilos(), rafina()])

    segundo = normalizar([kilos(), rafina()], tmp_path, OpenAIFalso(OTRAS_DECISIONES))

    assert segundo == primero


def test_una_segunda_ejecucion_con_los_mismos_insights_deja_los_dos_ficheros_iguales(tmp_path):
    primera_ejecucion(tmp_path, [kilos(), rafina()])
    antes = {nombre: leer(tmp_path, 12, nombre) for nombre in FICHEROS}

    normalizar([kilos(), rafina()], tmp_path, OpenAIFalso(OTRAS_DECISIONES))

    assert {nombre: leer(tmp_path, 12, nombre) for nombre in FICHEROS} == antes


def test_un_insight_nuevo_con_un_nombre_ya_resuelto_queda_resuelto_al_mismo_jugador(tmp_path):
    primera_ejecucion(tmp_path, [rafina()])

    resultado = normalizar(
        [rafina(), de_otro_video(rafina())], tmp_path, OpenAIFalso(OTRAS_DECISIONES)
    )

    assert [(i["fuente"]["video_id"], i["jugador_id"]) for i in resultado["insights"]] == [
        ("AT_UTIeNmtI", "411295"),
        ("OTRO_VIDEO1", "411295"),
    ]


def test_en_una_segunda_ejecucion_solo_se_consulta_el_nombre_nuevo(tmp_path):
    primera_ejecucion(tmp_path, [kilos(), rafina()])
    openai_falso = OpenAIFalso([("la mine", elige("937958"))])

    normalizar([kilos(), rafina(), insight("la mine", "la mine es duda")], tmp_path, openai_falso)

    assert ["la mine" in p["enviado"] for p in openai_falso.peticiones] == [True]


def primera_con_error_api(datos_dir):
    primera_ejecucion(datos_dir, [rafina()], [("Rafiña", openai.OpenAIError("fallo de conexión"))])


def test_un_nombre_eliminado_por_error_api_se_resuelve_en_la_siguiente_ejecucion(tmp_path):
    primera_con_error_api(tmp_path)

    normalizar([rafina()], tmp_path, OpenAIFalso([("Rafiña", elige("411295"))]))

    assert leer(tmp_path, 12, "insights_normalizados.json") == [{**rafina(), **RAPHINHA_RESUELTO}]


def test_un_nombre_eliminado_por_error_api_y_resuelto_despues_deja_de_estar_eliminado(tmp_path):
    primera_con_error_api(tmp_path)

    normalizar([rafina()], tmp_path, OpenAIFalso([("Rafiña", elige("411295"))]))

    assert leer(tmp_path, 12, "nombres_eliminados.json") == []


def test_si_el_identificador_guardado_ya_no_esta_en_la_lista_oficial_se_vuelve_a_consultar(
    tmp_path,
):
    primera_ejecucion(tmp_path, [rafina()])
    nueva_lista = [{**RAPHINHA, "id": "999001"}, LAMINE, GONZALO_MADRID, GONZALO_ESPANYOL]

    resultado = normalizar(
        [rafina()], tmp_path, OpenAIFalso([("Rafiña", elige("999001"))]), oficiales=nueva_lista
    )

    assert [i["jugador_id"] for i in resultado["insights"]] == ["999001"]


def test_un_nombre_eliminado_que_ya_no_llega_desaparece_de_nombres_eliminados(tmp_path):
    primera_ejecucion(tmp_path, [kilos(), rafina()])

    normalizar([rafina()], tmp_path, OpenAIFalso(DECISIONES_PRIMERA))

    assert leer(tmp_path, 12, "nombres_eliminados.json") == []


def test_un_insight_resuelto_que_ya_no_llega_desaparece_de_insights_normalizados(tmp_path):
    primera_ejecucion(tmp_path, [rafina("Cita 1"), rafina("Cita 2")])

    normalizar([rafina("Cita 1")], tmp_path, OpenAIFalso(DECISIONES_PRIMERA))

    assert [i["quote_original"] for i in leer(tmp_path, 12, "insights_normalizados.json")] == [
        "Cita 1"
    ]


FICHERO_DANADO = pytest.mark.parametrize(
    ("danado", "contenido"),
    [(nombre, contenido) for nombre in FICHEROS for contenido in ('[{"jugador":', "{}")],
    ids=[
        "insights_no_es_json",
        "insights_no_es_lista",
        "eliminados_no_es_json",
        "eliminados_no_es_lista",
    ],
)


def preparar_ficheros(datos_dir, danado, contenido):
    """Deja en la carpeta de la jornada 12 el fichero `danado` con `contenido` y el otro vacío."""
    carpeta = datos_dir / "2026" / "j12"
    carpeta.mkdir(parents=True)
    for nombre in FICHEROS:
        (carpeta / nombre).write_text(contenido if nombre == danado else "[]", encoding="utf-8")
    return carpeta


@FICHERO_DANADO
def test_un_fichero_guardado_danado_detiene_la_resolucion_con_un_error_que_dice_su_ruta(
    tmp_path, danado, contenido
):
    carpeta = preparar_ficheros(tmp_path, danado, contenido)

    with pytest.raises(Exception) as error:
        normalizar([rafina()], tmp_path, OpenAIFalso(DECISIONES_PRIMERA))

    assert str(carpeta / danado) in str(error.value)


@FICHERO_DANADO
def test_con_un_fichero_guardado_danado_no_se_hace_ninguna_peticion_al_modelo(
    tmp_path, danado, contenido
):
    preparar_ficheros(tmp_path, danado, contenido)
    openai_falso = OpenAIFalso(DECISIONES_PRIMERA)

    with pytest.raises(Exception):  # noqa: B017
        normalizar([rafina()], tmp_path, openai_falso)

    assert openai_falso.peticiones == []


@FICHERO_DANADO
def test_con_un_fichero_guardado_danado_los_dos_ficheros_quedan_como_estaban(
    tmp_path, danado, contenido
):
    carpeta = preparar_ficheros(tmp_path, danado, contenido)
    antes = {nombre: (carpeta / nombre).read_bytes() for nombre in FICHEROS}

    with pytest.raises(Exception):  # noqa: B017
        normalizar([rafina()], tmp_path, OpenAIFalso(DECISIONES_PRIMERA))

    assert {nombre: (carpeta / nombre).read_bytes() for nombre in FICHEROS} == antes


TCHOUAMENI = {
    "id": "413112",
    "nombre": "Aurélien Tchouaméni",
    "posicion": "Pivote",
    "equipo": "Real Madrid",
}
PEDRI = {"id": "683840", "nombre": "Pedri", "posicion": "Mediocentro", "equipo": "FC Barcelona"}
CON_PEDRI = [*OFICIALES, TCHOUAMENI, PEDRI]

# Las citas identifican cada consulta: los nombres pueden aparecer también en las
# instrucciones o entre los candidatos.
CITAS = {"Rafiña": "Primera cita", "Chamenamení": "Segunda cita", "Pedri": "Tercera cita"}


def tres_nombres():
    return [insight(nombre, cita) for nombre, cita in CITAS.items()]


def respuestas(chamenameni="413112", interrumpe_en=None, error_api_en=None):
    decisiones = {"Rafiña": "411295", "Chamenamení": chamenameni, "Pedri": "683840"}
    reglas = []
    for nombre, cita in CITAS.items():
        respuesta = elige(decisiones[nombre])
        if nombre == error_api_en:
            respuesta = openai.OpenAIError("fallo de conexión")
        if nombre == interrumpe_en:
            respuesta = KeyboardInterrupt()
        reglas.append((cita, respuesta))
    return reglas


def resolver_tres(datos_dir, openai_falso):
    return normalizar(tres_nombres(), datos_dir, openai_falso, oficiales=CON_PEDRI)


def interrumpida(datos_dir, interrumpe_en="Pedri", **cambios):
    try:
        resolver_tres(datos_dir, OpenAIFalso(respuestas(interrumpe_en=interrumpe_en, **cambios)))
    except KeyboardInterrupt:
        pass


def consultados(openai_falso):
    return [
        next((nombre for nombre, cita in CITAS.items() if cita in p["enviado"]), None)
        for p in openai_falso.peticiones
    ]


def test_si_se_interrumpe_en_el_tercer_nombre_quedan_guardados_los_insights_de_los_dos_primeros(
    tmp_path,
):
    interrumpida(tmp_path)

    guardados = leer(tmp_path, 12, "insights_normalizados.json")
    assert [(i["nombre_transcrito"], i["jugador_id"]) for i in guardados] == [
        ("Rafiña", "411295"),
        ("Chamenamení", "413112"),
    ]


def test_si_se_interrumpe_en_el_tercer_nombre_queda_guardado_el_eliminado_antes(tmp_path):
    interrumpida(tmp_path, chamenameni=None)

    assert leer(tmp_path, 12, "nombres_eliminados.json") == [
        {"nombre": "Chamenamení", "quote": "Segunda cita", "motivo": "sin_coincidencia"}
    ]


def test_tras_una_interrupcion_la_siguiente_ejecucion_solo_consulta_el_nombre_que_faltaba(
    tmp_path,
):
    interrumpida(tmp_path)
    openai_falso = OpenAIFalso(respuestas())

    resolver_tres(tmp_path, openai_falso)

    assert consultados(openai_falso) == ["Pedri"]


def test_reanudar_tras_una_interrupcion_da_lo_mismo_que_una_ejecucion_sin_interrumpir(tmp_path):
    interrumpida(tmp_path / "interrumpida", chamenameni=None)
    reanudado = resolver_tres(tmp_path / "interrumpida", OpenAIFalso(respuestas(chamenameni=None)))
    seguido = resolver_tres(tmp_path / "seguida", OpenAIFalso(respuestas(chamenameni=None)))

    def ficheros(datos_dir):
        return {nombre: leer(datos_dir, 12, nombre) for nombre in FICHEROS}

    assert (reanudado, ficheros(tmp_path / "interrumpida")) == (
        seguido,
        ficheros(tmp_path / "seguida"),
    )


def test_si_se_interrumpe_en_el_primer_nombre_los_ficheros_que_quedan_son_listas_json(tmp_path):
    interrumpida(tmp_path, interrumpe_en="Rafiña")

    carpeta = tmp_path / "2026" / "j12"
    existentes = [carpeta / nombre for nombre in FICHEROS if (carpeta / nombre).exists()]
    assert {
        fichero.name: isinstance(json.loads(fichero.read_text(encoding="utf-8")), list)
        for fichero in existentes
    } == {fichero.name: True for fichero in existentes}


def test_si_se_interrumpe_en_el_primer_nombre_la_siguiente_ejecucion_consulta_los_tres(tmp_path):
    interrumpida(tmp_path, interrumpe_en="Rafiña")
    openai_falso = OpenAIFalso(respuestas())

    resolver_tres(tmp_path, openai_falso)

    assert consultados(openai_falso) == ["Rafiña", "Chamenamení", "Pedri"]


def test_un_nombre_con_error_api_antes_de_la_interrupcion_se_vuelve_a_consultar(tmp_path):
    interrumpida(tmp_path, error_api_en="Rafiña")
    openai_falso = OpenAIFalso(respuestas())

    resolver_tres(tmp_path, openai_falso)

    assert consultados(openai_falso) == ["Rafiña", "Pedri"]
