import json
import re
from types import SimpleNamespace

import pytest

from fantasy_copilot.evaluacion import ErrorConjunto, evaluar_nombres

RAPHINHA = {
    "id": "411295",
    "nombre": "Raphinha",
    "posicion": "Extremo derecho",
    "equipo": "FC Barcelona",
}
PEDRI = {"id": "683840", "nombre": "Pedri", "posicion": "Mediocentro", "equipo": "FC Barcelona"}
TCHOUAMENI = {
    "id": "413112",
    "nombre": "Aurélien Tchouaméni",
    "posicion": "Pivote",
    "equipo": "Real Madrid",
}
CAMAVINGA = {
    "id": "640428",
    "nombre": "Eduardo Camavinga",
    "posicion": "Mediocentro",
    "equipo": "Real Madrid",
}
OFICIALES_2026 = [RAPHINHA, PEDRI, TCHOUAMENI, CAMAVINGA]

CITA_RAFINA = "Rafiña está en un momento brutal"
CITA_CHAMENAMENI = "Chamenamení vuelve al once"
CITA_PEDRI = "Pedri tiene que jugar sí o sí"
CITA_OK_BUENO = "ok bueno, vamos con el siguiente"
CITA_GAVI = "Gabi vuelve a entrenar"

_ESCAPE_UNICODE = re.compile(r"\\+u([0-9a-fA-F]{4})")


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
    aparece en los mensajes de la petición.
    """

    def __init__(self, reglas):
        self.reglas = reglas
        self.peticiones = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._crear))

    def _crear(self, **peticion):
        enviado = json.dumps(peticion.get("messages"), ensure_ascii=False, default=str)
        enviado = _ESCAPE_UNICODE.sub(lambda escape: chr(int(escape.group(1), 16)), enviado)
        self.peticiones.append(enviado)
        respuesta = next((r for marca, r in self.reglas if marca in enviado), None)
        if respuesta is None:
            raise AssertionError("Petición inesperada: ninguna regla coincide con su texto")
        return respuesta


def jugadores_de(temporada):
    return OFICIALES_2026 if temporada == 2026 else []


def escribir_conjunto(datos_dir, nombres, temporada=2026):
    fichero = datos_dir / "evaluacion" / "nombres.json"
    fichero.parent.mkdir(parents=True, exist_ok=True)
    fichero.write_text(
        json.dumps({"temporada": temporada, "nombres": nombres}, ensure_ascii=False),
        encoding="utf-8",
    )
    return fichero


def nombre(transcrito, cita, jugador_id):
    return {"nombre_transcrito": transcrito, "cita": cita, "jugador_id": jugador_id}


def evaluar(datos_dir, cliente):
    return evaluar_nombres(
        datos_dir=datos_dir, jugadores_de=jugadores_de, modelo="modelo-de-prueba", cliente=cliente
    )


CONJUNTO_DE_CUATRO = [
    nombre("Rafiña", CITA_RAFINA, RAPHINHA["id"]),
    nombre("Chamenamení", CITA_CHAMENAMENI, TCHOUAMENI["id"]),
    nombre("Pedri", CITA_PEDRI, PEDRI["id"]),
    nombre("Ok Bueno", CITA_OK_BUENO, None),
]


def openai_de_cuatro():
    return OpenAIFalso(
        [
            (CITA_RAFINA, elige(RAPHINHA["id"])),
            (CITA_CHAMENAMENI, elige(CAMAVINGA["id"])),
            (CITA_PEDRI, elige(None)),
            (CITA_OK_BUENO, elige(None)),
        ]
    )


def test_con_todos_los_nombres_resueltos_a_su_jugador_las_metricas_son_perfectas(tmp_path):
    escribir_conjunto(
        tmp_path,
        [
            nombre("Rafiña", CITA_RAFINA, RAPHINHA["id"]),
            nombre("Chamenamení", CITA_CHAMENAMENI, TCHOUAMENI["id"]),
            nombre("Pedri", CITA_PEDRI, PEDRI["id"]),
        ],
    )
    openai_falso = OpenAIFalso(
        [
            (CITA_RAFINA, elige(RAPHINHA["id"])),
            (CITA_CHAMENAMENI, elige(TCHOUAMENI["id"])),
            (CITA_PEDRI, elige(PEDRI["id"])),
        ]
    )

    resultado = evaluar(tmp_path, openai_falso)

    assert (
        resultado["precision"],
        resultado["cobertura"],
        resultado["porcentaje_eliminados"],
    ) == (1, 1, 0)


def test_precision_cobertura_y_eliminados_con_aciertos_errores_y_eliminados(tmp_path):
    escribir_conjunto(tmp_path, CONJUNTO_DE_CUATRO)

    resultado = evaluar(tmp_path, openai_de_cuatro())

    assert (
        resultado["precision"],
        resultado["cobertura"],
        resultado["porcentaje_eliminados"],
    ) == (0.5, pytest.approx(1 / 3), 0.5)


def test_el_resultado_detalla_por_cada_nombre_el_jugador_esperado_y_el_obtenido(tmp_path):
    escribir_conjunto(tmp_path, CONJUNTO_DE_CUATRO)

    resultado = evaluar(tmp_path, openai_de_cuatro())

    assert resultado["nombres"] == [
        {
            "nombre_transcrito": "Rafiña",
            "jugador_id_esperado": RAPHINHA["id"],
            "jugador_id_obtenido": RAPHINHA["id"],
        },
        {
            "nombre_transcrito": "Chamenamení",
            "jugador_id_esperado": TCHOUAMENI["id"],
            "jugador_id_obtenido": CAMAVINGA["id"],
        },
        {
            "nombre_transcrito": "Pedri",
            "jugador_id_esperado": PEDRI["id"],
            "jugador_id_obtenido": None,
        },
        {
            "nombre_transcrito": "Ok Bueno",
            "jugador_id_esperado": None,
            "jugador_id_obtenido": None,
        },
    ]


def test_un_nombre_que_espera_nulo_y_sale_resuelto_cuenta_como_resuelto_no_correcto(tmp_path):
    escribir_conjunto(tmp_path, [nombre("Ok Bueno", CITA_OK_BUENO, None)])
    openai_falso = OpenAIFalso([(CITA_OK_BUENO, elige(PEDRI["id"]))])

    resultado = evaluar(tmp_path, openai_falso)

    assert resultado["precision"] == 0


def test_sin_ningun_nombre_resuelto_la_precision_es_nula(tmp_path):
    escribir_conjunto(tmp_path, [nombre("Pedri", CITA_PEDRI, PEDRI["id"])])
    openai_falso = OpenAIFalso([(CITA_PEDRI, elige(None))])

    resultado = evaluar(tmp_path, openai_falso)

    assert resultado["precision"] is None


def test_si_ningun_nombre_espera_jugador_la_cobertura_es_nula(tmp_path):
    escribir_conjunto(tmp_path, [nombre("Ok Bueno", CITA_OK_BUENO, None)])
    openai_falso = OpenAIFalso([(CITA_OK_BUENO, elige(PEDRI["id"]))])

    resultado = evaluar(tmp_path, openai_falso)

    assert resultado["cobertura"] is None


def test_un_nombre_sin_cita_se_evalua_igual(tmp_path):
    escribir_conjunto(tmp_path, [{"nombre_transcrito": "Rafiña", "jugador_id": RAPHINHA["id"]}])
    openai_falso = OpenAIFalso([("", elige(RAPHINHA["id"]))])

    resultado = evaluar(tmp_path, openai_falso)

    assert resultado["nombres"] == [
        {
            "nombre_transcrito": "Rafiña",
            "jugador_id_esperado": RAPHINHA["id"],
            "jugador_id_obtenido": RAPHINHA["id"],
        }
    ]


def guardar_decisiones_en_todas_las_jornadas(datos_dir):
    """Decisiones previas contrarias a lo que responderá el modelo, en todas las jornadas."""
    for jornada in range(1, 39):
        directorio = datos_dir / "2026" / f"j{jornada:02d}"
        directorio.mkdir(parents=True)
        (directorio / "insights_normalizados.json").write_text(
            json.dumps(
                [
                    {"nombre_transcrito": "Rafiña", "jugador_id": PEDRI["id"]},
                    {"nombre_transcrito": "Chamenamení", "jugador_id": TCHOUAMENI["id"]},
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (directorio / "nombres_eliminados.json").write_text(
            json.dumps(
                [{"nombre": "Pedri", "quote": CITA_PEDRI, "motivo": "sin_coincidencia"}],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def test_se_consulta_al_modelo_por_cada_nombre_aunque_haya_decisiones_guardadas(tmp_path):
    guardar_decisiones_en_todas_las_jornadas(tmp_path)
    escribir_conjunto(
        tmp_path,
        [
            nombre("Rafiña", CITA_RAFINA, RAPHINHA["id"]),
            nombre("Chamenamení", CITA_CHAMENAMENI, TCHOUAMENI["id"]),
            nombre("Pedri", CITA_PEDRI, PEDRI["id"]),
        ],
    )
    openai_falso = OpenAIFalso(
        [
            (CITA_RAFINA, elige(RAPHINHA["id"])),
            (CITA_CHAMENAMENI, elige(CAMAVINGA["id"])),
            (CITA_PEDRI, elige(PEDRI["id"])),
        ]
    )

    resultado = evaluar(tmp_path, openai_falso)

    assert [n["jugador_id_obtenido"] for n in resultado["nombres"]] == [
        RAPHINHA["id"],
        CAMAVINGA["id"],
        PEDRI["id"],
    ]


def contenido_de(directorio):
    return {
        str(ruta.relative_to(directorio)): ruta.read_bytes() if ruta.is_file() else None
        for ruta in directorio.rglob("*")
    }


def test_evaluar_no_escribe_nada_en_las_carpetas_de_las_jornadas(tmp_path):
    guardar_decisiones_en_todas_las_jornadas(tmp_path)
    escribir_conjunto(
        tmp_path,
        [
            nombre("Gavi", CITA_GAVI, None),
            nombre("Rafiña", CITA_RAFINA, RAPHINHA["id"]),
        ],
    )
    antes = contenido_de(tmp_path / "2026")
    openai_falso = OpenAIFalso([(CITA_GAVI, elige(None)), (CITA_RAFINA, elige(RAPHINHA["id"]))])

    evaluar(tmp_path, openai_falso)

    assert contenido_de(tmp_path / "2026") == antes


def escribir_texto(datos_dir, texto):
    fichero = datos_dir / "evaluacion" / "nombres.json"
    fichero.parent.mkdir(parents=True, exist_ok=True)
    fichero.write_text(texto, encoding="utf-8")


CONJUNTOS_INVALIDOS = {
    "no es JSON": "{temporada: 2026,",
    "sin temporada": json.dumps({"nombres": [nombre("Rafiña", CITA_RAFINA, RAPHINHA["id"])]}),
    "sin nombres": json.dumps({"temporada": 2026}),
    "nombre sin nombre_transcrito": json.dumps(
        {"temporada": 2026, "nombres": [{"cita": CITA_RAFINA, "jugador_id": RAPHINHA["id"]}]}
    ),
    "nombre sin jugador_id": json.dumps(
        {"temporada": 2026, "nombres": [{"nombre_transcrito": "Rafiña", "cita": CITA_RAFINA}]}
    ),
}


@pytest.mark.parametrize("texto", CONJUNTOS_INVALIDOS.values(), ids=CONJUNTOS_INVALIDOS.keys())
def test_un_conjunto_mal_formado_lanza_un_error_que_nombra_el_fichero(tmp_path, texto):
    escribir_texto(tmp_path, texto)

    with pytest.raises(ErrorConjunto, match="nombres.json"):
        evaluar(tmp_path, OpenAIFalso([]))


def test_si_el_conjunto_no_existe_se_lanza_un_error_que_nombra_el_fichero(tmp_path):
    with pytest.raises(ErrorConjunto, match="nombres.json"):
        evaluar(tmp_path, OpenAIFalso([]))


def test_un_jugador_esperado_fuera_de_la_lista_oficial_lanza_un_error_que_nombra_el_nombre_oido(
    tmp_path,
):
    escribir_conjunto(
        tmp_path,
        [
            nombre("Rafiña", CITA_RAFINA, RAPHINHA["id"]),
            nombre("Lamín", "Lamín se sale", "999999"),
        ],
    )

    with pytest.raises(ErrorConjunto, match="Lamín"):
        evaluar(tmp_path, OpenAIFalso([(CITA_RAFINA, elige(RAPHINHA["id"]))]))
